# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Network to play no-press Diplomacy.

Comments referring to tensor shapes use the following abbreviations:
  B := batch size
  T := learners unroll length.
  PLAYERS := num players
  REP_SIZE := generic representation size for an entity.

  NUM_AREAS := observation_utils.NUM_AREAS
  NUM_PROVINCES := observation_utils.NUM_PROVINCES
  MAX_ACTION_INDEX := action_utils.MAX_ACTION_INDEX
  MAX_ORDERS := action_utils.MAX_ORDERS
"""

import collections
import functools
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

from diplomacy.environment import action_utils
from diplomacy.environment import observation_transformation
from diplomacy.environment import observation_utils as utils
from diplomacy.environment import province_order
from diplomacy.environment import tree_utils


def normalize_adjacency(adjacency: np.ndarray) -> np.ndarray:
  """Computes the symmetric normalized Laplacian of an adjacency matrix.

  Symmetric normalized Laplacians are the representation of choice for graphs in
  GraphConvNets (see https://arxiv.org/pdf/1609.02907.pdf).

  Args:
    adjacency: map adjacency matrix without self-connections.
  Returns:
    Symmetric normalized Laplacian matrix of adjacency.
  """
  adjacency += np.eye(*adjacency.shape)
  d = np.diag(np.power(adjacency.sum(axis=1), -0.5))
  return d.dot(adjacency).dot(d)


class EncoderCore(hk.Module):
  """Graph Network with non-shared weights across nodes.

  The input to this network is organized by area and the topology is described
  by the symmetric normalized Laplacian of an adjacency matrix.
  """

  def __init__(self,
               adjacency: jnp.ndarray,
               *,
               filter_size: int = 32,
               batch_norm_config: Optional[Dict[str, Any]] = None,
               name: str = "encoder_core"):
    """Constructor.

    Args:
      adjacency: [NUM_AREAS, NUM_AREAS] symmetric normalized Laplacian of the
          adjacency matrix.
      filter_size: output size of per-node linear layer.
      batch_norm_config: config dict for hk.BatchNorm.
      name: a name for the module.
    """
    super().__init__(name=name)
    self._adjacency = adjacency
    self._filter_size = filter_size
    bnc = dict(decay_rate=0.9, eps=1e-5, create_scale=True, create_offset=True)
    bnc.update(batch_norm_config or {})
    self._bn = hk.BatchNorm(**bnc)

  def __call__(self,
               tensors: jnp.ndarray,
               *,
               is_training: bool = False) -> jnp.ndarray:
    """One round of message passing.

    Output nodes are represented as the concatenation of the sum of incoming
    messages and the message sent.

    Args:
      tensors: [B, NUM_AREAS, REP_SIZE]
      is_training: Whether this is during training.

    Returns:
      [B, NUM_AREAS, 2 * self._filter_size]
    """
    w = hk.get_parameter(
        "w", shape=tensors.shape[-2:] + (self._filter_size,),
        init=hk.initializers.VarianceScaling())
    messages = jnp.einsum("bni,nij->bnj", tensors, w)
    tensors = jnp.matmul(self._adjacency, messages)
    tensors = jnp.concatenate([tensors, messages], axis=-1)
    tensors = self._bn(tensors, is_training=is_training)
    return jax.nn.relu(tensors)


class BoardEncoder(hk.Module):
  """Encode board state.

  Constructs a representation of the board state, organized per-area. The output
  depends on the season in the game, the specific power (player) we are
  considering as well as the number of builds for this player.

  Both season and player are embedded before being included in the
  representation.

  We first construct a "shared representation", which does not depend on the
  specific player, and then include player in the later layers.
  """

  def __init__(self,
               adjacency: jnp.ndarray,
               *,
               shared_filter_size: int = 32,
               player_filter_size: int = 32,
               num_shared_cores: int = 8,
               num_player_cores: int = 8,
               num_players: int = 7,
               num_seasons: int = utils.NUM_SEASONS,
               player_embedding_size: int = 16,
               season_embedding_size: int = 16,
               min_init_embedding: float = -1.0,
               max_init_embedding: float = 1.0,
               batch_norm_config: Optional[Dict[str, Any]] = None,
               name: str = "board_encoder"):
    """Constructor.

    Args:
      adjacency: [NUM_AREAS, NUM_AREAS] symmetric normalized Laplacian of the
        adjacency matrix.
      shared_filter_size: filter_size of each EncoderCore for shared layers.
      player_filter_size: filter_size of each EncoderCore for player-specific
        layers.
      num_shared_cores: number of shared layers, or rounds of message passing.
      num_player_cores: number of player-specific layers, or rounds of message
        passing.
      num_players: number of players.
      num_seasons: number of seasons.
      player_embedding_size: size of player embedding.
      season_embedding_size: size of season embedding.
      min_init_embedding: min value for hk.initializers.RandomUniform for player
        and season embedding.
      max_init_embedding: max value for hk.initializers.RandomUnifor for player
        and season embedding.
      batch_norm_config: config dict for hk.BatchNorm.
      name: a name for this module.
    """
    super().__init__(name=name)
    self._num_players = num_players
    self._season_embedding = hk.Embed(
        num_seasons,
        season_embedding_size,
        w_init=hk.initializers.RandomUniform(min_init_embedding,
                                             max_init_embedding))
    self._player_embedding = hk.Embed(
        num_players,
        player_embedding_size,
        w_init=hk.initializers.RandomUniform(min_init_embedding,
                                             max_init_embedding))
    make_encoder = functools.partial(
        EncoderCore,
        adjacency, batch_norm_config=batch_norm_config)
    self._shared_encode = make_encoder(filter_size=shared_filter_size)
    self._shared_core = [
        make_encoder(filter_size=shared_filter_size)
        for _ in range(num_shared_cores)
    ]
    self._player_encode = make_encoder(filter_size=player_filter_size)
    self._player_core = [
        make_encoder(filter_size=player_filter_size)
        for _ in range(num_player_cores)
    ]

    bnc = dict(decay_rate=0.9, eps=1e-5, create_scale=True, create_offset=True)
    bnc.update(batch_norm_config or {})
    self._bn = hk.BatchNorm(**bnc)

  def __call__(self,
               state_representation: jnp.ndarray,
               season: jnp.ndarray,
               build_numbers: jnp.ndarray,
               is_training: bool = False) -> jnp.ndarray:
    """Encoder board state.

    Args:
      state_representation: [B, NUM_AREAS, REP_SIZE].
      season: [B, 1].
      build_numbers: [B, 1].
      is_training: Whether this is during training.
    Returns:
      [B, NUM_AREAS, 2 * self._player_filter_size].
    """
    season_context = jnp.tile(
        self._season_embedding(season)[:, None], (1, utils.NUM_AREAS, 1))
    build_numbers = jnp.tile(build_numbers[:, None].astype(jnp.float32),
                             (1, utils.NUM_AREAS, 1))
    state_representation = jnp.concatenate(
        [state_representation, season_context, build_numbers], axis=-1)

    representation = self._shared_encode(
        state_representation, is_training=is_training)
    for layer in self._shared_core:
      representation += layer(representation, is_training=is_training)

    player_context = jnp.tile(
        self._player_embedding.embeddings[None, :, None, :],
        (season.shape[0], 1, utils.NUM_AREAS, 1))
    representation = jnp.tile(representation[:, None],
                              (1, self._num_players, 1, 1))
    representation = jnp.concatenate([representation, player_context], axis=3)
    representation = hk.BatchApply(self._player_encode)(
        representation, is_training=is_training)
    for layer in self._player_core:
      representation += hk.BatchApply(layer)(
          representation, is_training=is_training)
    return self._bn(representation, is_training=is_training)


class RecurrentOrderNetworkInput(NamedTuple):
  average_area_representation: jnp.ndarray
  legal_actions_mask: jnp.ndarray
  teacher_forcing: jnp.ndarray
  previous_teacher_forcing_action: jnp.ndarray
  temperature: jnp.ndarray


def previous_action_from_teacher_or_sample(
    teacher_forcing: jnp.ndarray,
    previous_teacher_forcing_action: jnp.ndarray,
    previous_sampled_action_index: jnp.ndarray):
  # Get previous action, from input (for teacher forcing) or state.
  return jnp.where(
      teacher_forcing, previous_teacher_forcing_action,
      jnp.asarray(action_utils.shrink_actions(
          action_utils.POSSIBLE_ACTIONS))[previous_sampled_action_index])


def one_hot_provinces_for_all_actions():
  return jax.nn.one_hot(
      jnp.asarray(action_utils.ordered_province(action_utils.POSSIBLE_ACTIONS)),
      utils.NUM_PROVINCES)


def blocked_provinces_and_actions(
    previous_action: jnp.ndarray,
    previous_blocked_provinces: jnp.ndarray):
  """Calculate which provinces and actions are illegal."""

  # Compute which provinces are blocked by past decisions.
  updated_blocked_provinces = jnp.maximum(
      previous_blocked_provinces, ordered_provinces_one_hot(previous_action))

  blocked_actions = jnp.squeeze(
      jnp.matmul(one_hot_provinces_for_all_actions(),
                 updated_blocked_provinces[..., None]), axis=-1)
  blocked_actions *= jnp.logical_not(
      jnp.asarray(is_waive(action_utils.POSSIBLE_ACTIONS)))
  return updated_blocked_provinces, blocked_actions


def sample_from_logits(
    logits: jnp.ndarray,
    legal_action_mask: jnp.ndarray,
    temperature: jnp.ndarray,):
  """Sample from logits respecting a legal actions mask."""
  deterministic_logits = jnp.where(
      jax.nn.one_hot(
          jnp.argmax(logits, axis=-1),
          num_classes=action_utils.MAX_ACTION_INDEX,
          dtype=jnp.bool_), 0,
      jnp.finfo(jnp.float32).min)
  stochastic_logits = jnp.where(legal_action_mask,
                                logits / temperature,
                                jnp.finfo(jnp.float32).min)

  logits_for_sampling = jnp.where(
      jnp.equal(temperature, 0.0),
      deterministic_logits,
      stochastic_logits)

  # Sample an action for the current province and update the state so that
  # following orders can be conditioned on this decision.
  key = hk.next_rng_key()
  return jax.random.categorical(
      key, logits_for_sampling, axis=-1)


class RelationalOrderDecoderState(NamedTuple):
  prev_orders: jnp.ndarray
  blocked_provinces: jnp.ndarray
  sampled_action_index: jnp.ndarray


class RelationalOrderDecoder(hk.RNNCore):
  """RelationalOrderDecoder.

  Relational Order Decoders (ROD)s output order logits for a unit, based on the
  current board representation, and the orders selected for other units so far.
  """

  def __init__(self,
               adjacency: jnp.ndarray,
               *,
               filter_size: int = 32,
               num_cores: int = 4,
               batch_norm_config: Optional[Dict[str, Any]] = None,
               name: str = "relational_order_decoder"):
    """Constructor.

    Args:
      adjacency: [NUM_PROVINCES, NUM_PROVINCES] symmetric normalized Laplacian
        of the per-province adjacency matrix.
      filter_size: filter_size for relational cores
      num_cores: number of relational cores
      batch_norm_config: config  dict for hk.BatchNorm
      name: module's name.
    """
    super().__init__(name=name)
    self._filter_size = filter_size
    self._encode = EncoderCore(
        adjacency,
        filter_size=self._filter_size,
        batch_norm_config=batch_norm_config)
    self._cores = []
    for _ in range(num_cores):
      self._cores.append(
          EncoderCore(
              adjacency,
              filter_size=self._filter_size,
              batch_norm_config=batch_norm_config))
    self._projection_size = 2 * self._filter_size  # (nodes, messages)

    bnc = dict(decay_rate=0.9, eps=1e-5, create_scale=True, create_offset=True)
    bnc.update(batch_norm_config or {})
    self._bn = hk.BatchNorm(**bnc)

  def _scatter_to_province(self, vector: jnp.ndarray,
                           scatter: jnp.ndarray) -> jnp.ndarray:
    """Scatters vector to its province location in inputs.

    Args:
      vector: [B*PLAYERS, REP_SIZE]
      scatter: [B*PLAYER, NUM_PROVINCES] -- one-hot encoding.

    Returns:
      [B*PLAYERS, NUM_AREAS, REP_SIZE] where vectors has been added in the
      location prescribed by scatter.
    """
    return vector[:, None, :] * scatter[..., None]

  def _gather_province(self, inputs: jnp.ndarray,
                       gather: jnp.ndarray) -> jnp.ndarray:
    """Gathers specific province location from inputs.

    Args:
      inputs: [B*PLAYERS, NUM_PROVINCES, REP_SIZE]
      gather: [B*PLAYERS, NUM_PROVINCES] -- one-hot encoding

    Returns:
      [B*PLAYERS, REP_SIZE] gathered from inputs.
    """
    return jnp.sum(inputs * gather[..., None], axis=1)

  def _relational_core(self,
                       previous_orders: jnp.ndarray,
                       board_representation,
                       is_training: bool = False):
    """Apply relational core to current province and previous decisions."""

    inputs = jnp.concatenate([previous_orders, board_representation], axis=-1)
    representation = self._encode(inputs, is_training=is_training)
    for core in self._cores:
      representation += core(representation, is_training=is_training)
    return self._bn(representation, is_training=is_training)

  def __call__(
      self,
      inputs: RecurrentOrderNetworkInput,
      prev_state: RelationalOrderDecoderState,
      *,
      is_training: bool = False,
  ) -> Tuple[jnp.ndarray, RelationalOrderDecoderState]:
    """Issue an order based on board representation and previous decisions.

    Args:
      inputs: RecurrentOrderNetworkInput(
          average_area_representation <-- [B*PLAYERS, REP_SIZE],
          legal_actions_mask <-- [B*PLAYERS, MAX_ACTION_INDEX],
          teacher_forcing <-- [B*PLAYERS],
          previous_teacher_forcing_action <-- [B*PLAYERS],
          temperature <-- [B*PLAYERS, 1]
        )
      prev_state: RelationalOrderDecoderState(
          prev_orders <-- [B*PLAYERS, NUM_PROVINCES, 2 * self._filter_size],
          blocked_provinces <-- [B*PLAYERS, NUM_PROVINCES],
          sampled_action_index <-- [B*PLAYER]
      )
      is_training: Whether this is during training.
    Returns:
      logits with shape [B*PLAYERS, MAX_ACTION_INDEX],
      updated RelationalOrderDecoderState shapes as above
    """

    projection = hk.get_parameter(
        "projection",
        shape=(action_utils.MAX_ACTION_INDEX, self._projection_size),
        init=hk.initializers.VarianceScaling())

    previous_action = previous_action_from_teacher_or_sample(
        inputs.teacher_forcing,
        inputs.previous_teacher_forcing_action,
        prev_state.sampled_action_index,)

    updated_blocked_provinces, blocked_actions = blocked_provinces_and_actions(
        previous_action, prev_state.blocked_provinces)

    # Construct representation of previous order.
    previous_order_representation = (previous_action[:, None] > 0) * projection[
        previous_action >> (action_utils.ACTION_INDEX_START - 32)]

    legal_actions_provinces = (
        jnp.matmul(inputs.legal_actions_mask,
                   one_hot_provinces_for_all_actions()) > 0)

    # Place the representation of the province currently under consideration in
    # the appropriate slot in the graph.
    scattered_board_representation = jnp.array(0.0) + self._scatter_to_province(
        inputs.average_area_representation,
        legal_actions_provinces)

    # Place previous order in the appropriate slot in the graph.
    scattered_previous_orders = (
        prev_state.prev_orders + self._scatter_to_province(
            previous_order_representation,
            jax.nn.one_hot(
                ordered_provinces(previous_action),
                utils.NUM_PROVINCES)))

    # Construct order logits conditional on province representation and previous
    # orders.
    board_representation = self._relational_core(
        scattered_previous_orders,
        scattered_board_representation,
        is_training=is_training)
    province_representation = self._gather_province(
        board_representation, legal_actions_provinces)
    order_logits = jnp.matmul(province_representation, projection.T)

    # Eliminate illegal actions
    is_legal_action = inputs.legal_actions_mask * (blocked_actions == 0)
    order_logits = jnp.where(is_legal_action, order_logits,
                             jnp.finfo(jnp.float32).min)

    action_index = sample_from_logits(order_logits, is_legal_action,
                                      inputs.temperature)

    return order_logits, RelationalOrderDecoderState(
        prev_orders=scattered_previous_orders,
        blocked_provinces=updated_blocked_provinces,
        sampled_action_index=action_index
    )

  def initial_state(
      self,
      batch_size: int,
      dtype: np.dtype = jnp.float32) -> RelationalOrderDecoderState:
    return RelationalOrderDecoderState(
        prev_orders=jnp.zeros(
            shape=(batch_size, utils.NUM_PROVINCES, 2 * self._filter_size),
            dtype=dtype),
        blocked_provinces=jnp.zeros(
            shape=(batch_size, utils.NUM_PROVINCES), dtype=dtype),
        sampled_action_index=jnp.zeros(
            shape=(batch_size,), dtype=jnp.int32),
    )


def ordered_provinces(actions: jnp.ndarray):
  return jnp.bitwise_and(
      jnp.right_shift(actions, action_utils.ACTION_ORDERED_PROVINCE_START),
      (1 << action_utils.ACTION_PROVINCE_BITS) - 1)


def is_waive(actions: jnp.ndarray):
  return jnp.equal(jnp.bitwise_and(
      jnp.right_shift(actions, action_utils.ACTION_ORDER_START),
      (1 << action_utils.ACTION_ORDER_BITS) - 1), action_utils.WAIVE)


def loss_from_logits(logits, actions, discounts):
  """Returns cross-entropy loss, unless actions are None; then it's entropy."""
  if actions is not None:
    action_indices = actions >> (action_utils.ACTION_INDEX_START - 32)
    loss = jnp.take_along_axis(
        -jax.nn.log_softmax(logits), action_indices[..., None],
        axis=-1).squeeze(-1)
    # Only look at loss for actual actions.
    loss = jnp.where(actions > 0, loss, 0)
  else:
    loss = (jax.nn.softmax(logits) * -jax.nn.log_softmax(logits)).sum(-1)
  loss = loss.sum(3)
  # Only look at adequate players.
  loss *= discounts
  return loss.mean()


def ordered_provinces_one_hot(actions, dtype=jnp.float32):
  provinces = jax.nn.one_hot(
      action_utils.ordered_province(actions), utils.NUM_PROVINCES, dtype=dtype)
  provinces *= ((actions > 0) & ~action_utils.is_waive(actions)).astype(
      dtype)[..., None]
  return provinces


def reorder_actions(actions, areas, season):
  """Reorder actions to match area ordering."""
  area_provinces = jax.nn.one_hot(
      [utils.province_id_and_area_index(a)[0] for a in range(utils.NUM_AREAS)],
      utils.NUM_PROVINCES,
      dtype=jnp.float32)
  provinces = jnp.tensordot(areas.astype(jnp.float32), area_provinces,
                            (-1, 0)).astype(jnp.int32)
  action_provinces = ordered_provinces_one_hot(actions, jnp.int32)
  ordered_actions = jnp.sum(
      jnp.sum(actions[..., None] * action_provinces, -2, keepdims=True) *
      provinces, -1)
  n_actions_found = jnp.sum(
      jnp.sum(action_provinces, -2, keepdims=True) * provinces, -1)
  # `actions` has `-1`s for missing actions.
  ordered_actions += n_actions_found - 1
  is_build = jnp.equal(season[..., None, None], utils.Season.BUILDS.value)
  tile_multiples = (1, 1) + actions.shape[2:]
  skip_reorder = jnp.tile(is_build, tile_multiples)
  reordered_actions = jnp.where(skip_reorder, actions, ordered_actions)

  return reordered_actions


class Network(hk.Module):
  """Policy and Value Networks for Diplomacy.

  This network processes the board state to produce action and values for a full
  turn of Diplomacy.

  In Diplomacy, at each turn, all players submit orders for all the units they
  control. This Network outputs orders for each unit, one by one. Orders for
  later units depend on decisoins made previously. We organize this as follows:

    shared_inference: computations shared by all units (e.g. encode board).
    initial_inference: set up initial state to implement inter-unit dependence.
    step_inference: compute order for one unit.
    inference: full turn inference (organizes other methods in obvious ways).
  """

  @classmethod
  def initial_inference_params_and_state(
      cls, constructor_kwargs, rng, num_players):
    def _inference(observations):
      network = cls(**constructor_kwargs)  # pytype: disable=not-instantiable
      return network.inference(observations)

    inference_fn = hk.transform_with_state(_inference)

    params, net_state = inference_fn.init(
        rng, tree_utils.tree_expand_dims(
            cls.get_observation_transformer(constructor_kwargs
                                            ).zero_observation(num_players)))
    return params, net_state

  @classmethod
  def get_observation_transformer(cls, class_constructor_kwargs, rng_key=None):
    del class_constructor_kwargs  # Unused
    return observation_transformation.GeneralObservationTransformer(
        rng_key=rng_key)

  @classmethod
  def zero_observation(cls, class_constructor_kwargs, num_players):
    return cls.get_observation_transformer(
        class_constructor_kwargs).zero_observation(num_players)

  def __init__(
      self,
      *,
      rnn_ctor,
      rnn_kwargs,
      name: str = "delta",
      num_players: int = 7,
      area_mdf: province_order.MapMDF = province_order.MapMDF.BICOASTAL_MAP,
      province_mdf: province_order.MapMDF = province_order.MapMDF.STANDARD_MAP,
      is_training: bool = False,
      shared_filter_size: int = 32,
      player_filter_size: int = 32,
      num_shared_cores: int = 8,
      num_player_cores: int = 8,
      value_mlp_hidden_layer_sizes: Sequence[int] = (256,),
      actions_since_last_moves_embedding_size: int = 10,
      batch_norm_config: Optional[Dict[str, Any]] = None,
  ):
    """Constructor.

    Args:
      rnn_ctor: Constructor for the RNN. The RNN will be constructed as
        `rnn_ctor(batch_norm_config=batch_norm_config, **rnn_kwargs)`.
      rnn_kwargs: kwargs for the RNN.
      name: a name for this module.
      num_players: number of players in the game, usually 7 (standard Diplomacy)
        or 2 (1v1 Diplomacy).
      area_mdf: path to mdf file containing a description of the board organized
        by area (e.g. Spain, Spain North Coast and Spain South Coast).
      province_mdf: path to mdf file containing a description of the board
        organized by province (e.g. Spain).
      is_training: whether this is a training instance.
      shared_filter_size: filter size in BoardEncoder, shared (across players)
        layers.
      player_filter_size: filter size in BoardEncoder, player specific layers.
      num_shared_cores: depth of BoardEncoder, shared (across players) layers.
      num_player_cores: depth of BoardEncoder, player specific layers.
      value_mlp_hidden_layer_sizes: sizes for value head. Output layer with size
        num_players is appended by this module.
      actions_since_last_moves_embedding_size: embedding size for last moves
        actions.
      batch_norm_config: kwargs for batch norm, eg the cross_replica_axis.
    """
    super().__init__()
    self._area_adjacency = normalize_adjacency(
        province_order.build_adjacency(
            province_order.get_mdf_content(area_mdf)))
    self._province_adjacency = normalize_adjacency(
        province_order.build_adjacency(
            province_order.get_mdf_content(province_mdf)))
    self._is_training = is_training

    self._moves_actions_encoder = hk.Embed(
        action_utils.MAX_ACTION_INDEX + 1,
        actions_since_last_moves_embedding_size)
    self._board_encoder = BoardEncoder(
        self._area_adjacency,
        shared_filter_size=shared_filter_size,
        player_filter_size=player_filter_size,
        num_shared_cores=num_shared_cores,
        num_player_cores=num_player_cores,
        batch_norm_config=batch_norm_config)
    self._last_moves_encoder = BoardEncoder(
        self._area_adjacency,
        shared_filter_size=shared_filter_size,
        player_filter_size=player_filter_size,
        num_shared_cores=num_shared_cores,
        num_player_cores=num_player_cores,
        batch_norm_config=batch_norm_config)
    self._rnn = rnn_ctor(batch_norm_config=batch_norm_config, **rnn_kwargs)
    self._value_mlp = hk.nets.MLP(
        output_sizes=list(value_mlp_hidden_layer_sizes) + [num_players])

  def loss_info(
      self,
      step_types: jnp.ndarray,  # [B,T+1].
      rewards: jnp.ndarray,  # [B,T+1].
      discounts: jnp.ndarray,  # [B,T+1].
      observations: Tuple[  # Batch dimensions [B,T+1].
          Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray],
      step_outputs: Dict[str, Any]  # Batch dimensions [B, T].
  ) -> Dict[str, jnp.ndarray]:
    """Losses to update the network's policy given a batch of experience.

    `(step_type[i], rewards[i], observations[i])` is the `actions[i]`. In
    other words, the reward accumulated over this sequence is
    `sum(rewards[1:])`, obtained by taking actions `actions`. The
    observation at the end of the sequence (resulting from
    `step_outputs.actions[-1]`) is `observations[-1]`.

    Args:
      step_types: tensor of step types.
      rewards: tensor of rewards.
      discounts: tensor of discounts.
      observations: observations, in format given by observation_transform. This
        is a tuple (initial_observation, step_observations, num_actions). Within
        a turn step_observations are stacked for each step.
      step_outputs: tensor of network outputs produced by inference.

    Returns:
      dict of logging outputs, including per-batch per-timestep losses.
    """
    del step_types, rewards  # unused
    observations = tree.map_structure(lambda xs: xs[:, :-1], observations)
    discounts = discounts[:, :-1]
    returns = step_outputs["returns"]
    initial_observation, step_observations, sequence_lengths = observations

    # Losses are always built with temperature 1.0
    step_observations["temperature"] = jnp.ones_like(
        step_observations["temperature"])

    # Reorder the actions to match with the legal_actions ordering.
    actions = reorder_actions(
        step_outputs["actions"],
        season=initial_observation["season"],
        areas=step_observations["areas"])

    # Get all but the last order for teacher forcing.
    last_action = actions[..., :-1]

    # Pad the previous action with -1 for the first order, as that is
    # never forced.
    last_action = jnp.concatenate(
        [-jnp.ones_like(last_action[:, :, :, :1]), last_action], axis=3)
    # Set teacher forcing actions.
    step_observations["last_action"] = last_action
    (initial_outputs, step_outputs) = hk.BatchApply(
        functools.partial(self.inference, all_teacher_forcing=True))(
            (initial_observation, step_observations, sequence_lengths))

    policy_logits = step_outputs["logits"]
    value_logits = initial_outputs["value_logits"]

    policy_loss = loss_from_logits(policy_logits, actions, discounts)
    policy_entropy = loss_from_logits(policy_logits, None, discounts)
    value_loss = -(jax.nn.log_softmax(value_logits) *
                   returns).sum(-1) * jnp.max(discounts, -1)
    returns_entropy = -(
        jnp.sum(returns * jnp.log(returns + 1e-7), -1) * jnp.max(discounts, -1))
    loss = policy_loss + value_loss

    # Get accuracy.
    labels = (
        jnp.asarray(action_utils.shrink_actions(
            action_utils.POSSIBLE_ACTIONS)) == actions[..., None])
    greedy_prediction = jnp.argmax(policy_logits, axis=-1)
    gathered_labels = jnp.take_along_axis(
        labels.astype(jnp.float32), greedy_prediction[..., None],
        -1).squeeze(-1)
    accuracy = jnp.count_nonzero(
        gathered_labels, axis=(2, 3)).astype(jnp.float32)
    accuracy_weights = jnp.count_nonzero(
        labels, axis=(2, 3, 4)).astype(jnp.float32)
    accuracy = jnp.sum(accuracy) / jnp.sum(accuracy_weights)

    # Get full-move accuracy.
    nonempty = jnp.any(labels, (-1, -2))
    whole_correct = jnp.equal(
        jnp.count_nonzero(gathered_labels, -1),
        jnp.count_nonzero(labels, (-1, -2)))
    whole_accuracy = jnp.count_nonzero(whole_correct
                                       & nonempty, 2).astype(jnp.float32)
    whole_accuracy_weights = jnp.count_nonzero(
        nonempty, axis=2).astype(jnp.float32)
    whole_accuracy = jnp.sum(whole_accuracy) / jnp.sum(whole_accuracy_weights)

    # For comparison, calculate the loss from a uniform random agent.
    ur_policy_logits = jnp.finfo(
        jnp.float32).min * (1 - step_observations["legal_actions_mask"])
    ur_policy_logits = jnp.broadcast_to(ur_policy_logits, policy_logits.shape)
    ur_policy_loss = loss_from_logits(ur_policy_logits, actions, discounts)
    ur_value_loss = -(jax.nn.log_softmax(jnp.zeros_like(value_logits)) *
                      returns).sum(-1) * jnp.max(discounts, -1)

    loss_dict = {
        "policy_loss": policy_loss,
        "policy_entropy": policy_entropy,
        "value_loss": value_loss,
        "total_loss": loss,
        "returns_entropy": returns_entropy,
        "uniform_random_policy_loss": ur_policy_loss,
        "uniform_random_value_loss": ur_value_loss,
        "uniform_random_total_loss": ur_value_loss + ur_policy_loss,
        "accuracy": accuracy,
        "accuracy_weight": jnp.sum(accuracy_weights),
        "whole_accuracy": whole_accuracy,
        "whole_accuracy_weight": jnp.sum(whole_accuracy_weights),
    }
    return tree.map_structure(jnp.mean, loss_dict)

  def loss(self, step_types, rewards, discounts, observations,
           step_outputs) -> jnp.ndarray:
    """Imitation learning loss."""
    losses = self.loss_info(step_types, rewards, discounts, observations,
                            step_outputs)
    return losses["total_loss"]

  def shared_rep(
      self, initial_observation: Dict[str, jnp.ndarray]
  ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """Processing shared by all units that require an order.

    Encodes board state, season and previous moves and implements.
    Computes value head.

    Args:
      initial_observation: see initial_observation_spec.

    Returns:
      value information and shared board representation.
    """
    # Stitch together board current situation and past moves.
    season = initial_observation["season"]
    build_numbers = initial_observation["build_numbers"]
    board = initial_observation["board_state"]

    last_moves = initial_observation["last_moves_phase_board_state"]

    moves_actions = jnp.sum(
        self._moves_actions_encoder(
            1 + initial_observation["actions_since_last_moves_phase"]),
        axis=-2)
    last_moves = jnp.concatenate([last_moves, moves_actions], axis=-1)

    # Compute board representation.

    # [B, PLAYERS, NUM_AREAS, REP_SIZE]
    board_representation = self._board_encoder(
        board, season, build_numbers, is_training=self._is_training)
    last_moves_representation = self._last_moves_encoder(
        last_moves, season, build_numbers, is_training=self._is_training)
    area_representation = jnp.concatenate(
        [board_representation, last_moves_representation], axis=-1)

    # Compute value head.
    value_logits = self._value_mlp(jnp.mean(area_representation, axis=(1, 2)))
    return (collections.OrderedDict(
        value_logits=value_logits,
        values=jax.nn.softmax(value_logits)), area_representation)

  def initial_inference(
      self, shared_rep: jnp.ndarray, player: jnp.ndarray
  ) -> Tuple[Dict[str, jnp.ndarray], Any]:
    """Set up initial state to implement inter-unit dependence."""
    batch_size = shared_rep.shape[0]
    return (jax.vmap(functools.partial(jnp.take, axis=0))(shared_rep,
                                                          player.squeeze(1)),
            self._rnn.initial_state(batch_size=batch_size))

  def step_inference(
      self,
      step_observation: Dict[str, jnp.ndarray],
      inference_internal_state: Any,
      all_teacher_forcing: bool = False
  ) -> Tuple[Dict[str, jnp.ndarray], Tuple[jnp.ndarray,
                                           Any]]:
    """Computes logits for 1 unit that requires order.

    Args:
      step_observation: see step_observation_spec.
      inference_internal_state: Board representation for each player and
        RelationalOrderDecoder previous state.
      all_teacher_forcing: Whether to leave sampled actions out of the
        inference state (for a learning speed boost).

    Returns:
      action information for this unit, updated inference_internal_state
    """
    area_representation, rnn_state = inference_internal_state
    average_area_representation = jnp.matmul(
        step_observation["areas"][:, None].astype(np.float32),
        area_representation).squeeze(1) / utils.NUM_AREAS
    inputs = RecurrentOrderNetworkInput(
        average_area_representation=average_area_representation,
        legal_actions_mask=step_observation["legal_actions_mask"],
        teacher_forcing=step_observation["last_action"] != 0,
        previous_teacher_forcing_action=step_observation["last_action"],
        temperature=step_observation["temperature"],
    )
    logits, updated_rnn_state = self._rnn(
        inputs, rnn_state, is_training=self._is_training)
    policy = jax.nn.softmax(logits)
    legal_action_mask = logits > jnp.finfo(jnp.float32).min

    actions = jnp.take_along_axis(
        jnp.asarray(action_utils.shrink_actions(
            action_utils.POSSIBLE_ACTIONS))[None],
        updated_rnn_state.sampled_action_index[..., None],
        axis=1).squeeze(1)
    if all_teacher_forcing:
      updated_rnn_state = updated_rnn_state._replace(
          sampled_action_index=jnp.zeros_like(
              updated_rnn_state.sampled_action_index))
    next_inference_internal_state = (area_representation, updated_rnn_state)
    return collections.OrderedDict(
        actions=actions,
        legal_action_mask=legal_action_mask,
        policy=policy,
        logits=logits), next_inference_internal_state

  def inference(
      self,
      observation: Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray],
                         jnp.ndarray],
      num_copies_each_observation: Optional[Tuple[int]] = None,
      all_teacher_forcing: bool = False,
  ) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Computes value estimates and actions for the full turn.

    Args:
      observation: see observation_spec.
      num_copies_each_observation: How many times to copy each observation. This
        allows us to produce multiple samples for the same state without
        recalculating the deterministic part of the Network.
      all_teacher_forcing: Whether to leave sampled actions out of the
        inference state (for a learning speed boost).

    Returns:
      Value and action information for the full turn.
    """
    initial_observation, step_observations, seq_lengths = observation
    num_players = int(seq_lengths.shape[1])
    initial_outputs, shared_rep = self.shared_rep(initial_observation)

    initial_inference_states_list = []
    batch_dim = jnp.shape(seq_lengths)[0]
    for player in range(num_players):
      player_tensor = jnp.full((1, 1), player)
      player_tensor = jnp.tile(player_tensor, (batch_dim, 1))
      initial_inference_states_list.append(
          self.initial_inference(shared_rep, player_tensor))

    initial_inference_states = tree.map_structure(
        lambda *x: jnp.stack(x, axis=1), *initial_inference_states_list)

    rnn_inputs = (step_observations, seq_lengths, initial_inference_states)
    # Replicate the inputs to the RNN according to num_copies_each_observation.
    if num_copies_each_observation is not None:
      num_copies = np.array(num_copies_each_observation)
      rnn_inputs = tree.map_structure(
          lambda x: jnp.repeat(x, num_copies, axis=0), rnn_inputs)

    def _apply_rnn_one_player(
        player_step_observations,  # [B, 17, ...]
        player_sequence_length,  # [B]
        player_initial_state):  # [B]

      player_step_observations = tree.map_structure(jnp.asarray,
                                                    player_step_observations)

      def apply_one_step(state, i):
        output, next_state = self.step_inference(
            tree.map_structure(lambda x: x[:, i], player_step_observations),
            state,
            all_teacher_forcing=all_teacher_forcing)

        def update(x, y, i=i):
          return jnp.where(
              i >= player_sequence_length[np.s_[:,] + (None,) * (x.ndim - 1)],
              x, y)

        state = tree.map_structure(update, state, next_state)
        zero_output = tree.map_structure(jnp.zeros_like, output)
        return state, tree.map_structure(update, zero_output, output)

      _, outputs = hk.scan(apply_one_step, player_initial_state,
                           jnp.arange(action_utils.MAX_ORDERS))
      return tree.map_structure(lambda x: x.swapaxes(0, 1), outputs)

    outputs = hk.BatchApply(_apply_rnn_one_player)(*rnn_inputs)

    return initial_outputs, outputs

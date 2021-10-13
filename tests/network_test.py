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

"""Tests basic functionality of the networks defined in network.py."""

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import numpy as np
import optax
import tree

from diplomacy.environment import action_utils
from diplomacy.environment import observation_utils as utils
from diplomacy.environment import province_order
from diplomacy.network import network


def _random_adjacency_matrix(num_nodes: int) -> np.ndarray:
  adjacency = np.random.randint(0, 2, size=(num_nodes, num_nodes))
  adjacency = adjacency + adjacency.T
  adjacency = np.clip(adjacency, 0, 1)
  adjacency[np.diag_indices(adjacency.shape[0])] = 0
  return network.normalize_adjacency(
      adjacency.astype(np.float32))


def test_network_rod_kwargs(filter_size=8, is_training=True):
  province_adjacency = network.normalize_adjacency(
      province_order.build_adjacency(
          province_order.get_mdf_content(province_order.MapMDF.STANDARD_MAP)))
  rod_kwargs = dict(
      adjacency=province_adjacency,
      filter_size=filter_size,
      num_cores=2,)

  network_kwargs = dict(
      rnn_ctor=network.RelationalOrderDecoder,
      rnn_kwargs=rod_kwargs,
      is_training=is_training,
      shared_filter_size=filter_size,
      player_filter_size=filter_size,
      num_shared_cores=2,
      num_player_cores=2,
      value_mlp_hidden_layer_sizes=[filter_size]
  )
  return network_kwargs


class NetworkTest(parameterized.TestCase):

  @parameterized.named_parameters([
      ('training', True),
      ('testing', False),
  ])
  @hk.testing.transform_and_run
  def test_encoder_core(self, is_training):

    batch_size = 10
    num_nodes = 5
    input_size = 4
    filter_size = 8
    expected_output_size = 2 * filter_size  # concat(edges, nodes)
    adjacency = _random_adjacency_matrix(num_nodes)

    tensors = np.random.randn(batch_size, num_nodes, input_size)

    model = network.EncoderCore(
        adjacency=adjacency, filter_size=filter_size)

    if not is_training:
      # ensure moving averages are created
      model(tensors, is_training=True)
    output_tensors = model(tensors, is_training=is_training)

    self.assertTupleEqual(output_tensors.shape,
                          (batch_size, num_nodes, expected_output_size))

  @parameterized.named_parameters([
      ('training', True),
      ('testing', False),
  ])
  @hk.testing.transform_and_run
  def test_board_encoder(self, is_training):

    batch_size = 10
    input_size = 4
    filter_size = 8
    num_players = 7
    expected_output_size = 2 * filter_size  # concat(edges, nodes)

    adjacency = _random_adjacency_matrix(utils.NUM_AREAS)

    state_representation = np.random.randn(batch_size, utils.NUM_AREAS,
                                           input_size)
    season = np.random.randint(0, utils.NUM_SEASONS, size=(batch_size,))
    build_numbers = np.random.randint(0, 5, size=(batch_size, num_players))

    model = network.BoardEncoder(
        adjacency=adjacency,
        player_filter_size=filter_size,
        num_players=num_players,
        num_seasons=utils.NUM_SEASONS)
    if not is_training:
      model(
          state_representation, season, build_numbers,
          is_training=True)  # ensure moving averages are created
    output_tensors = model(
        state_representation, season, build_numbers, is_training=is_training)

    self.assertTupleEqual(
        output_tensors.shape,
        (batch_size, num_players, utils.NUM_AREAS, expected_output_size))

  @parameterized.named_parameters([
      ('training', True),
      ('testing', False),
  ])
  @hk.testing.transform_and_run
  def test_relational_order_decoder(self, is_training):
    batch_size = 10
    num_players = 7
    adjacency = _random_adjacency_matrix(utils.NUM_PROVINCES)
    relational_order_decoder = network.RelationalOrderDecoder(
        adjacency)
    input_sequence = network.RecurrentOrderNetworkInput(
        average_area_representation=np.zeros(
            shape=(batch_size * num_players, action_utils.MAX_ORDERS, 64),
            dtype=np.float32),
        legal_actions_mask=np.zeros(
            shape=(batch_size * num_players, action_utils.MAX_ORDERS,
                   action_utils.MAX_ACTION_INDEX),
            dtype=np.uint8),
        teacher_forcing=np.zeros(
            shape=(batch_size * num_players, action_utils.MAX_ORDERS),
            dtype=bool),
        previous_teacher_forcing_action=np.zeros(
            shape=(batch_size * num_players, action_utils.MAX_ORDERS),
            dtype=np.int32),
        temperature=np.zeros(
            shape=(batch_size * num_players, action_utils.MAX_ORDERS, 1),
            dtype=np.float32),
    )
    state = relational_order_decoder.initial_state(batch_size=batch_size *
                                                   num_players)

    if not is_training:
      relational_order_decoder(
          tree.map_structure(lambda s: s[:, 0], input_sequence),
          state,
          is_training=True)  # ensure moving averages are created
    for t in range(action_utils.MAX_ORDERS):
      outputs, state = relational_order_decoder(
          tree.map_structure(
              lambda s: s[:, t],  # pylint: disable=cell-var-from-loop
              input_sequence),
          state,
          is_training=is_training)
    self.assertTupleEqual(
        outputs.shape,
        (batch_size * num_players, action_utils.MAX_ACTION_INDEX))

  @hk.testing.transform_and_run
  def test_shared_rep(self):
    batch_size = 10
    num_players = 7
    filter_size = 8
    expected_output_size = (4 * filter_size
                           )  # (edges + nodes) * (board + alliance)
    network_kwargs = test_network_rod_kwargs(filter_size=filter_size)
    net = network.Network(**network_kwargs)
    initial_observations, _, _ = network.Network.zero_observation(
        network_kwargs, num_players=num_players)
    batched_initial_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], batch_size, axis=0),
        initial_observations)
    value, representation = net.shared_rep(batched_initial_observations)
    self.assertTupleEqual(value['value_logits'].shape,
                          (batch_size, num_players))
    self.assertTupleEqual(value['values'].shape, (batch_size, num_players))
    self.assertTupleEqual(
        representation.shape,
        (batch_size, num_players, utils.NUM_AREAS, expected_output_size))

  @hk.testing.transform_and_run
  def test_inference(self):
    batch_size = 2
    copies = [2, 3]

    num_players = 7
    network_kwargs = test_network_rod_kwargs()
    net = network.Network(**network_kwargs)

    observations = network.Network.zero_observation(
        network_kwargs, num_players=num_players)
    batched_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], batch_size, axis=0), observations)
    initial_outputs, step_outputs = net.inference(batched_observations, copies)
    self.assertTupleEqual(initial_outputs['values'].shape,
                          (batch_size, num_players))
    self.assertTupleEqual(initial_outputs['value_logits'].shape,
                          (batch_size, num_players))
    self.assertTupleEqual(step_outputs['actions'].shape,
                          (sum(copies), num_players, action_utils.MAX_ORDERS))
    self.assertTupleEqual(step_outputs['legal_action_mask'].shape,
                          (sum(copies), num_players, action_utils.MAX_ORDERS,
                           action_utils.MAX_ACTION_INDEX))
    self.assertTupleEqual(step_outputs['policy'].shape,
                          (sum(copies), num_players, action_utils.MAX_ORDERS,
                           action_utils.MAX_ACTION_INDEX))
    self.assertTupleEqual(step_outputs['logits'].shape,
                          (sum(copies), num_players, action_utils.MAX_ORDERS,
                           action_utils.MAX_ACTION_INDEX))

  @hk.testing.transform_and_run
  def test_loss_info(self):
    batch_size = 4
    time_steps = 2
    num_players = 7
    network_kwargs = test_network_rod_kwargs()
    net = network.Network(**network_kwargs)

    observations = network.Network.zero_observation(
        network_kwargs, num_players=num_players)
    sequence_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], time_steps + 1, axis=0), observations)
    batched_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], batch_size, axis=0),
        sequence_observations)

    rewards = np.zeros(
        shape=(batch_size, time_steps + 1, num_players), dtype=np.float32)
    discounts = np.zeros(
        shape=(batch_size, time_steps + 1, num_players), dtype=np.float32)
    actions = np.zeros(
        shape=(batch_size, time_steps, num_players, action_utils.MAX_ORDERS),
        dtype=np.int32)
    returns = np.zeros(
        shape=(batch_size, time_steps, num_players), dtype=np.float32)
    loss_info = net.loss_info(
        step_types=None,
        rewards=rewards,
        discounts=discounts,
        observations=batched_observations,
        step_outputs={
            'actions': actions,
            'returns': returns
        })
    expected_keys = [
        'policy_loss',
        'policy_entropy',
        'value_loss',
        'total_loss',
        'returns_entropy',
        'uniform_random_policy_loss',
        'uniform_random_value_loss',
        'uniform_random_total_loss',
        'accuracy',
        'accuracy_weight',
        'whole_accuracy',
        'whole_accuracy_weight',
    ]
    self.assertSetEqual(set(loss_info), set(expected_keys))
    for value in loss_info.values():
      self.assertTupleEqual(value.shape, tuple())

  def test_inference_not_is_training(self):
    """Tests inferring with is_training set to False."""
    batch_size = 4
    time_steps = 2
    num_players = 7

    network_kwargs = test_network_rod_kwargs()
    observations = network.Network.zero_observation(
        network_kwargs, num_players=num_players)

    sequence_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], time_steps + 1, axis=0), observations)
    batched_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], batch_size, axis=0),
        sequence_observations)
    rewards = np.zeros(
        shape=(batch_size, time_steps + 1, num_players), dtype=np.float32)
    discounts = np.zeros(
        shape=(batch_size, time_steps + 1, num_players), dtype=np.float32)
    actions = np.zeros(
        shape=(batch_size, time_steps, num_players, action_utils.MAX_ORDERS),
        dtype=np.int32)
    returns = np.zeros(
        shape=(batch_size, time_steps, num_players), dtype=np.float32)
    rng = jax.random.PRNGKey(42)
    step_outputs = {'actions': actions, 'returns': returns}

    def _loss_info(unused_step_types, rewards, discounts, observations,
                   step_outputs):
      net = network.Network(**test_network_rod_kwargs())
      return net.loss_info(None, rewards, discounts, observations, step_outputs)

    # Get parameters from a training network.
    loss_module = hk.transform_with_state(_loss_info)
    params, loss_state = loss_module.init(rng, None, rewards, discounts,
                                          batched_observations, step_outputs)

    def inference(observations, num_copies_each_observation):
      net = network.Network(**test_network_rod_kwargs(is_training=False))
      return net.inference(observations, num_copies_each_observation)

    # Do inference on a test time network.
    inference_module = hk.transform_with_state(inference)
    inference_module.apply(params, loss_state, rng, sequence_observations, None)

  def test_take_gradients(self):
    """Test applying a gradient update step."""
    batch_size = 4
    time_steps = 2
    num_players = 7

    network_kwargs = test_network_rod_kwargs()
    observations = network.Network.zero_observation(
        network_kwargs, num_players=num_players)
    sequence_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], time_steps + 1, axis=0), observations)
    batched_observations = tree.map_structure(
        lambda x: np.repeat(x[None, ...], batch_size, axis=0),
        sequence_observations)
    rewards = np.zeros(
        shape=(batch_size, time_steps + 1, num_players), dtype=np.float32)
    discounts = np.zeros(
        shape=(batch_size, time_steps + 1, num_players), dtype=np.float32)
    actions = np.zeros(
        shape=(batch_size, time_steps, num_players, action_utils.MAX_ORDERS),
        dtype=np.int32)
    returns = np.zeros(
        shape=(batch_size, time_steps, num_players), dtype=np.float32)
    rng = jax.random.PRNGKey(42)
    step_outputs = {'actions': actions, 'returns': returns}

    def _loss_info(unused_step_types, rewards, discounts, observations,
                   step_outputs):
      net = network.Network(**test_network_rod_kwargs())
      return net.loss_info(None, rewards, discounts, observations, step_outputs)

    loss_module = hk.transform_with_state(_loss_info)
    params, loss_state = loss_module.init(rng, None, rewards, discounts,
                                          batched_observations, step_outputs)

    def _loss(params, state, rng, rewards, discounts, observations,
              step_outputs):
      losses, state = loss_module.apply(params, state, rng, None, rewards,
                                        discounts, observations, step_outputs)
      total_loss = losses['total_loss'].mean()
      return total_loss, (losses, state)

    (_, (_, loss_state)), grads = jax.value_and_grad(
        _loss, has_aux=True)(params, loss_state, rng, rewards, discounts,
                             batched_observations, step_outputs)

    opt_init, opt_update = optax.adam(0.001)
    opt_state = opt_init(params)
    updates, opt_state = opt_update(grads, opt_state)
    params = optax.apply_updates(params, updates)


if __name__ == '__main__':
  absltest.main()

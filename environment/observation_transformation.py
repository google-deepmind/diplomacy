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

"""Functions for observation transformation and base classes for their users."""

import collections
import enum
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

from dm_env import specs
import jax.numpy as jnp
import numpy as np
import tree

from diplomacy.environment import action_utils
from diplomacy.environment import observation_utils as utils
from diplomacy.environment import province_order
from diplomacy.environment import tree_utils


class ObservationTransformState(NamedTuple):
  # Board state at the last moves phase.
  previous_board_state: np.ndarray
  # Most recent board state.
  last_action_board_state: np.ndarray
  # Actions taken since the last moves phase.
  actions_since_previous_moves_phase: np.ndarray
  # Records if last phase was a moves phase.
  last_phase_was_moves: bool


def update_state(
    observation: utils.Observation,
    prev_state: Optional[ObservationTransformState]
    ) -> ObservationTransformState:
  """Returns an updated state for alliance features."""
  if prev_state is None:
    last_phase_was_moves = False
    last_board_state = None
    previous_board_state = np.zeros(shape=utils.OBSERVATION_BOARD_SHAPE)
    actions_since_previous_moves_phase = np.full((utils.NUM_AREAS, 3),
                                                 -1,
                                                 dtype=np.int32)
  else:
    (previous_board_state, last_board_state, actions_since_previous_moves_phase,
     last_phase_was_moves) = prev_state
  actions_since_previous_moves_phase = actions_since_previous_moves_phase.copy()

  if last_phase_was_moves:
    actions_since_previous_moves_phase[:] = -1
    last_phase_was_moves = False

  actions_since_previous_moves_phase[:] = np.roll(
      actions_since_previous_moves_phase, axis=1, shift=-1)
  actions_since_previous_moves_phase[:, -1] = -1

  for action in observation.last_actions:
    order_type, (province_id, coast), _, _ = action_utils.action_breakdown(
        action)
    if order_type == action_utils.WAIVE:
      continue
    elif order_type == action_utils.BUILD_ARMY:
      area = utils.area_from_province_id_and_area_index(province_id, 0)
    elif order_type == action_utils.BUILD_FLEET:
      if utils.obs_index_start_and_num_areas(province_id)[1] == 3:
        area = utils.area_from_province_id_and_area_index(
            province_id, coast + 1)
      else:
        area = utils.area_from_province_id_and_area_index(province_id, 0)
    else:
      area = utils.area_id_for_unit_in_province_id(province_id,
                                                   last_board_state)
    assert actions_since_previous_moves_phase[area, -1] == -1
    actions_since_previous_moves_phase[area, -1] = action >> 48

  if observation.season.is_moves():
    previous_board_state = observation.board
    # On the next season, update the alliance features.
    last_phase_was_moves = True
  return ObservationTransformState(previous_board_state,
                                   observation.board,
                                   actions_since_previous_moves_phase,
                                   last_phase_was_moves)


class TopologicalIndexing(enum.Enum):
  NONE = 0
  MILA = 1

MILA_TOPOLOGICAL_ORDER = [
    'YOR', 'EDI', 'LON', 'LVP', 'NTH', 'WAL', 'CLY', 'NWG', 'ECH', 'IRI', 'NAO',
    'BEL', 'DEN', 'HEL', 'HOL', 'NWY', 'SKA', 'BAR', 'BRE', 'MAO', 'PIC', 'BUR',
    'RUH', 'BAL', 'KIE', 'SWE', 'FIN', 'STP', 'STP/NC', 'GAS', 'PAR', 'NAF',
    'POR', 'SPA', 'SPA/NC', 'SPA/SC', 'WES', 'MAR', 'MUN', 'BER', 'GOB', 'LVN',
    'PRU', 'STP/SC', 'MOS', 'TUN', 'GOL', 'TYS', 'PIE', 'BOH', 'SIL', 'TYR',
    'WAR', 'SEV', 'UKR', 'ION', 'TUS', 'NAP', 'ROM', 'VEN', 'GAL', 'VIE', 'TRI',
    'ARM', 'BLA', 'RUM', 'ADR', 'AEG', 'ALB', 'APU', 'EAS', 'GRE', 'BUD', 'SER',
    'ANK', 'SMY', 'SYR', 'BUL', 'BUL/EC', 'CON', 'BUL/SC']

mila_topological_index = province_order.topological_index(
        province_order.get_mdf_content(province_order.MapMDF.BICOASTAL_MAP),
        MILA_TOPOLOGICAL_ORDER)


class GeneralObservationTransformer:
  """A general observation transformer class.

  Additional fields should default to False to avoid changing existing
  configs. Additional arguments to the obs transform functions that support
  optional fields must be keyword-only arguments.
  """

  def __init__(
      self,
      *,
      rng_key: Optional[jnp.ndarray],
      board_state: bool = True,
      last_moves_phase_board_state: bool = True,
      actions_since_last_moves_phase: bool = True,
      season: bool = True,
      build_numbers: bool = True,
      topological_indexing: TopologicalIndexing = TopologicalIndexing.NONE,
      areas: bool = True,
      last_action: bool = True,
      legal_actions_mask: bool = True,
      temperature: bool = True,
  ) -> None:
    """Constructor which configures the fields the transformer will return.

    Each argument represents whether a particular field should be included in
    the observation, except for topological indexing.

    Args:
      rng_key: A Jax random number generator key, for use if an observation
        transformation is ever stochastic.
      board_state: Flag for whether to include the current board state,
        an array containing current unit positions, dislodged units, supply
        centre ownership, and where units may be removed or built.
      last_moves_phase_board_state: Flag for whether to include the board state
        at the start of the last moves phase. If actions_since_last_moves is
        True, this board state is necessary to give context to the actions.
      actions_since_last_moves_phase: Flag for whether to include the actions
        since the last moves phase. These are given by area, with 3 channels
        (for moves, retreats and builds phases). If there was no action in the
        area, then the field is a 0.
      season: Flag for whether to include the current season in the observation.
        There are five seasons, as listed in observation_utils.Season.
      build_numbers: Flag for whether to include the number of builds/disbands
        each player has. Always 0 except in a builds phase.
      topological_indexing: When choosing unit actions in sequence, the order
        they are chosen is determined by the order step_observations sort the
        areas by. This config determines that ordering. NONE orders them
        according to the area index in the observation, MILA uses the same
        ordering as in Pacquette et al.
      areas: Flag for whether to include a vector of length NUM_AREAS, which is
        True in the area that the next unit-action will be chosen for.
      last_action: Flag for whether to include the action chosen in the previous
        unit-action selection in the input. This is used e.g. by teacher
        forcing. When sampling from a network, it can use the sample it drew in
        the previous step of the policy head.
      legal_actions_mask: Flag for whether to include a mask of which actions
        are legal. It will be based on the consecutive action indexes, and have
        length constants.MAX_ACTION_INDEX.
      temperature: Flag for whether to include a sampling temperature in the
        neural network input.
    """
    self._rng_key = rng_key
    self.board_state = board_state
    self.last_moves_phase_board_state = last_moves_phase_board_state
    self.actions_since_last_moves_phase = actions_since_last_moves_phase
    self.season = season
    self.build_numbers = build_numbers
    self.areas = areas
    self.last_action = last_action
    self.legal_actions_mask = legal_actions_mask
    self.temperature = temperature

    self._topological_indexing = topological_indexing

  def initial_observation_spec(
      self,
      num_players: int
  ) -> Dict[str, specs.Array]:
    """Returns a spec for the output of initial_observation_transform."""
    spec = collections.OrderedDict()

    if self.board_state:
      spec['board_state'] = specs.Array(
          (utils.NUM_AREAS, utils.PROVINCE_VECTOR_LENGTH), dtype=np.float32)

    if self.last_moves_phase_board_state:
      spec['last_moves_phase_board_state'] = specs.Array(
          (utils.NUM_AREAS, utils.PROVINCE_VECTOR_LENGTH), dtype=np.float32)

    if self.actions_since_last_moves_phase:
      spec['actions_since_last_moves_phase'] = specs.Array(
          (utils.NUM_AREAS, 3), dtype=np.int32)

    if self.season:
      spec['season'] = specs.Array((), dtype=np.int32)

    if self.build_numbers:
      spec['build_numbers'] = specs.Array((num_players,), dtype=np.int32)

    return spec

  def initial_observation_transform(
      self,
      observation: utils.Observation,
      prev_state: Optional[ObservationTransformState]
  ) -> Tuple[Dict[str, jnp.ndarray], ObservationTransformState]:
    """Constructs initial Network observations and state.

    See initial_observation_spec for array sizes, and the README for details on
    how to construct each field.
    Please implement your observation_test to check that these are constructed
    properly.

    Args:
      observation: Parsed observation from environment
      prev_state: previous ObservationTransformState

    Returns:
      initial observations and inital state.

    """
    next_state = update_state(observation, prev_state)

    initial_observation = collections.OrderedDict()

    if self.board_state:
      initial_observation['board_state'] = np.array(observation.board,
                                                    dtype=np.float32)

    if self.last_moves_phase_board_state:
      initial_observation['last_moves_phase_board_state'] = np.array(
          prev_state.previous_board_state if prev_state else
          observation.board, dtype=np.float32)

    if self.actions_since_last_moves_phase:
      initial_observation['actions_since_last_moves_phase'] = np.cast[np.int32](
          next_state.actions_since_previous_moves_phase)

    if self.season:
      initial_observation['season'] = np.cast[np.int32](
          observation.season.value)

    if self.build_numbers:
      initial_observation['build_numbers'] = np.array(
          observation.build_numbers, dtype=np.int32)

    return initial_observation, next_state

  def step_observation_spec(
      self
  ) -> Dict[str, specs.Array]:
    """Returns a spec for the output of step_observation_transform."""
    spec = collections.OrderedDict()

    if self.areas:
      spec['areas'] = specs.Array(shape=(utils.NUM_AREAS,), dtype=bool)

    if self.last_action:
      spec['last_action'] = specs.Array(shape=(), dtype=np.int32)

    if self.legal_actions_mask:
      spec['legal_actions_mask'] = specs.Array(
          shape=(action_utils.MAX_ACTION_INDEX,), dtype=np.uint8)

    if self.temperature:
      spec['temperature'] = specs.Array(shape=(1,), dtype=np.float32)

    return spec

  def step_observation_transform(
      self,
      transformed_initial_observation: Dict[str, jnp.ndarray],
      legal_actions: Sequence[jnp.ndarray],
      slot: int,
      last_action: int,
      area: int,
      step_count: int,
      previous_area: Optional[int],
      temperature: float
  ) -> Dict[str, jnp.ndarray]:
    """Converts raw step obs. from the diplomacy env. to network inputs.

    See step_observation_spec for array sizes, and the README for details on
    how to construct each field.
    Please implement your observation_test to check that these are constructed
    properly.

    Args:
      transformed_initial_observation: Initial observation made with same config
      legal_actions: legal actions for all players this turn
      slot: the slot/player_id we are creating the obs for
      last_action: the player's last action (used for teacher forcing)
      area: the area to create an action for
      step_count: how many unit actions have been created so far
      previous_area: the area for the previous unit action
      temperature: the sampling temperature for unit actions

    Returns:
      The step observation.
    """
    del previous_area  # Unused
    # Areas to sum over.
    areas = np.zeros(shape=(utils.NUM_AREAS,), dtype=bool)
    if area == utils.INVALID_AREA_FLAG:
      raise NotImplementedError('network requires area ordering to be '
                                'specified')
    if area == utils.BUILD_PHASE_AREA_FLAG:
      build_numbers = transformed_initial_observation['build_numbers']
      board = transformed_initial_observation['board_state']
      legal_actions_list = legal_actions[slot]
      if build_numbers[slot] > 0:
        player_areas = utils.build_areas(slot, board)
      else:
        player_areas = utils.removable_areas(slot, board)
      areas[player_areas] = True
    else:
      province, _ = utils.province_id_and_area_index(area)
      legal_actions_list = action_utils.actions_for_province(
          legal_actions[slot], province)
      areas[area] = True

    if not legal_actions_list:
      raise ValueError('No legal actions found for area {}'.format(area))
    legal_actions_mask = np.full(action_utils.MAX_ACTION_INDEX, False)
    legal_actions_mask[action_utils.action_index(
        np.array(legal_actions_list))] = True

    step_obs = collections.OrderedDict()

    if self.areas:
      step_obs['areas'] = areas

    if self.last_action:
      step_obs['last_action'] = np.array(
          action_utils.shrink_actions(last_action if step_count else -1),
          dtype=np.int32)

    if self.legal_actions_mask:
      step_obs['legal_actions_mask'] = legal_actions_mask

    if self.temperature:
      step_obs['temperature'] = np.array([temperature], dtype=np.float32)

    return step_obs

  def observation_spec(
      self,
      num_players: int
  ) -> Tuple[Dict[str, specs.Array], Dict[str, specs.Array], specs.Array]:
    """Returns a spec for the output of observation_transform."""
    return (
        self.initial_observation_spec(num_players),  # Initial
        tree.map_structure(
            lambda x: x.replace(  # pylint: disable=g-long-lambda
                shape=(num_players, action_utils.MAX_ORDERS) + x.shape),
            self.step_observation_spec()),  # Step Observations
        specs.Array((num_players,), dtype=np.int32))  # Sequence Lengths

  def zero_observation(self, num_players):
    return tree.map_structure(lambda spec: spec.generate_value(),
                              self.observation_spec(num_players))

  def observation_transform(
      self,
      *,
      observation: utils.Observation,
      legal_actions: Sequence[np.ndarray],
      slots_list: Sequence[int],
      prev_state: Any,
      temperature: float,
      area_lists: Optional[Sequence[Sequence[int]]] = None,
      forced_actions: Optional[Sequence[Sequence[int]]] = None,
  ) -> Tuple[Tuple[Dict[str, jnp.ndarray],
                   Dict[str, jnp.ndarray], Sequence[int]],
             ObservationTransformState]:
    """Transform the observation into the format required by Network policies.

    Args:
      observation: Observation from environment
      legal_actions: legal actions for all players this turn
      slots_list: the slots/player_ids we are creating obs for
      prev_state: previous ObservationTransformState
      temperature: the sampling temperature for unit actions
      area_lists: Order to process areas in. None for a default ordering.
      forced_actions: actions from teacher forcing. None when sampling.

    Returns:
      (initial_observation, stacked_step_observations,
      step_observation_sequence_lengths), next_obs_transform_state
    """
    if area_lists is None:
      area_lists = []
      for player in slots_list:
        topo_index = self._topological_index()
        area_lists.append(
            utils.order_relevant_areas(observation, player, topo_index))

    initial_observation, next_state = self.initial_observation_transform(
        observation, prev_state)
    num_players = len(legal_actions)
    sequence_lengths = np.zeros(shape=(num_players,), dtype=np.int32)
    zero_step_obs = tree.map_structure(
        specs.Array.generate_value,
        self.step_observation_spec()
    )

    step_observations = [[zero_step_obs] * action_utils.MAX_ORDERS
                         for _ in range(num_players)]

    if len(slots_list) != len(area_lists):
      raise ValueError('area_lists and slots_list different lengths')

    for player, area_list in zip(
        slots_list, area_lists):
      sequence_lengths[player] = len(area_list)
      previous_area = utils.INVALID_AREA_FLAG  # No last action on 1st iteration
      for i, area in enumerate(area_list):
        last_action = 0
        if forced_actions is not None and i > 0:
          # Find the right last action, in case the forced actions are not in
          # the order this network produces actions.
          if area in (utils.INVALID_AREA_FLAG, utils.BUILD_PHASE_AREA_FLAG):
            last_action = forced_actions[player][i - 1]
          else:
            # Find the action with the right area.
            last_action = action_utils.find_action_with_area(
                forced_actions[player], previous_area)
        step_observations[player][i] = self.step_observation_transform(
            initial_observation, legal_actions, player, last_action, area, i,
            previous_area, temperature)
        previous_area = area

    stacked_step_obs_per_player = []
    for player in range(num_players):
      stacked_step_obs_per_player.append(
          tree_utils.tree_stack(step_observations[player]))

    stacked_step_obs = tree_utils.tree_stack(stacked_step_obs_per_player)

    return (initial_observation, stacked_step_obs, sequence_lengths), next_state

  def _topological_index(self):
    """Returns the order in which to produce orders from different areas.

    If None, the order in the observation will be used.

    Returns:
      A list of areas
    Raises:
      RuntimeError: on hitting unexpected branch
    """
    if self._topological_indexing == TopologicalIndexing.NONE:
      return None
    elif self._topological_indexing == TopologicalIndexing.MILA:
      return mila_topological_index
    else:
      raise RuntimeError('Unexpected Branch')

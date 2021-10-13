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

"""Play games of Diplomacy."""

from typing import Any, Dict, List, Optional, Sequence

from absl import logging
import numpy as np

from diplomacy.environment import action_utils
from diplomacy.environment import observation_utils as utils
from diplomacy.network import network_policy


class DiplomacyTrajectory:
  """Stores data from a Diplomacy game."""

  def __init__(self):
    self.observations: List[utils.Observation] = []
    self.legal_actions: List[np.ndarray] = []
    self.actions: List[np.ndarray] = []
    self.step_outputs: List[Dict[str, Any]] = []
    self.returns: Optional[np.ndarray] = None

  def append_step(self,
                  observation: utils.Observation,
                  legal_actions: np.ndarray,
                  actions: np.ndarray,
                  step_outputs: Dict[str, Any]):
    self.observations.append(observation)
    self.legal_actions.append(legal_actions)
    self.actions.append(actions)
    self.step_outputs.append(step_outputs)

  def terminate(self, returns):
    self.returns = returns


def _draw_returns(
    points_per_supply_centre: bool,
    board: np.ndarray,
    num_players: int
) -> np.ndarray:
  """Computes returns (number of supply centers) when the game end in a draw."""
  if points_per_supply_centre:
    returns = [len(utils.sc_provinces(i, board)) for i in range(num_players)]
  else:
    returns = [
        1 if utils.sc_provinces(i, board) else 0 for i in range(num_players)]
  return np.array(returns, dtype=np.float32) / sum(returns)


def run_game(
    state,
    policies: Sequence[network_policy.Policy],
    slots_to_policies: Sequence[int],
    max_length: Optional[int] = None,
    min_years_forced_draw=1000,
    forced_draw_probability=0.0,
    points_per_supply_centre=False,
    draw_if_slot_loses=None,
) -> DiplomacyTrajectory:
  """Run a game of diplomacy.

  Args:
    state: A DiplomacyState in Spring 1901 (see diplomacy_state.py).
    policies: sequence of policies which are acting.
    slots_to_policies: sequence of length num_players mapping slots of the
      games to the index of the corresponding policy in policies.
    max_length: terminate games after this many full diplomacy turns.
    min_years_forced_draw: minimum years to consider force a draw.
    forced_draw_probability: probability of a draw each year after the first
      min_years_forced_draw
    points_per_supply_centre: whether to assign points per supply centre in a
      draw (rather than 0/1 for win/loss).
    draw_if_slot_loses: if this slot is eliminated, the game is ended in a draw.

  Returns:
    Trajectory of the game, as a DiplomacyTrajectory.
  """
  num_players = 7

  if len(slots_to_policies) != num_players:
    raise ValueError(
        f"Length of slot to policy mapping {len(slots_to_policies)}"
        f", but {num_players} players in game.")
  policies_to_slots_lists = [[] for i in range(len(policies))]
  for slot in range(num_players):
    policy_index = slots_to_policies[slot]
    if policy_index >= len(policies) or policy_index < 0:
      raise ValueError(f"Policy index {policy_index} out of range")
    policies_to_slots_lists[policy_index].append(slot)

  for policy in policies:
    policy.reset()

  if max_length is None:
    max_length = np.inf

  year = 0
  turn_num = 0

  traj = DiplomacyTrajectory()
  returns = None

  while not state.is_terminal() and turn_num < max_length:
    logging.info("In turn %d year %d ", turn_num, year)

    observation = state.observation()

    # New Game Year Checks
    if observation.season == utils.Season.SPRING_MOVES:
      year += 1
      if (draw_if_slot_loses is not None and
          not utils.sc_provinces(draw_if_slot_loses, observation.board)):
        returns = _draw_returns(points_per_supply_centre, observation.board,
                                num_players)
        logging.info("Forcing a draw due to elimination - returns %s",
                     returns)
        break
      if (year > min_years_forced_draw and
          np.random.uniform() < forced_draw_probability):
        returns = _draw_returns(points_per_supply_centre, observation.board,
                                num_players)
        logging.info("Forcing a draw at year %s - returns %s", year, returns)
        break

    legal_actions = state.legal_actions()
    padded_legal_actions = np.zeros(
        (num_players, action_utils.MAX_LEGAL_ACTIONS), np.int64)
    for i in range(num_players):
      padded_legal_actions[i, :len(legal_actions[i])] = legal_actions[i]
    actions_lists = [[] for _ in range(num_players)]
    policies_step_outputs = {}

    for policy, slots_list in zip(policies, policies_to_slots_lists):
      (policy_actions_lists,
       policies_step_outputs[str(policy)]) = policy.actions(
           slots_list, observation, legal_actions)
      if len(policy_actions_lists) != len(slots_list):
        raise ValueError(f"Policy {policy} returned {len(policy_actions_lists)}"
                         f" actions lists for {len(slots_list)} players")
      for actions, slot in zip(policy_actions_lists, slots_list):
        actions_lists[slot] = actions
    # Save our actions lists.
    padded_actions = np.full(
        (num_players, action_utils.MAX_ORDERS), -1, np.int64)
    for i, actions_list in enumerate(actions_lists):
      if actions_list is not None:
        padded_actions[i, :len(actions_list)] = actions_list

    state.step(actions_lists)
    turn_num += 1

    traj.append_step(observation,
                     padded_legal_actions,
                     padded_actions,
                     policies_step_outputs)

  if returns is None:
    returns = state.returns()
  traj.terminate(returns)

  return traj

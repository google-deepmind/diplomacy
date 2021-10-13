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

"""Utilities for parsing actions and observations produced by the environment."""
from typing import Sequence, Tuple, Union
import numpy as np

from diplomacy.environment import action_list
from diplomacy.environment import observation_utils as utils

# Actions are represented using 64 bit integers. In particular bits 0-31 contain
# the action order, represented using the following format:
# ORDER|ORDERED PROVINCE|TARGET PROVINCE|THIRD PROVINCE, (each of these takes up
# to 8 bits). Bits 32-47 are always 0. Bits 48-63 are used to record the index
# of each action into POSSIBLE_ACTIONS (see action_list.py).
# A detailed explanation of this format is provided in the README file.

# Order is in bits 0 to 7.
ACTION_ORDER_START = 0
ACTION_ORDER_BITS = 8
ACTION_PROVINCE_BITS = 7
# Ordered unit's province is in bits 9 to 15.
ACTION_ORDERED_PROVINCE_START = 9
# Ordered unit's province coast is in bit 8.
ACTION_ORDERED_PROVINCE_COAST = 8
# Target (for support hold, support move, move, convoy) is in bits 17 to 23
ACTION_TARGET_PROVINCE_START = 17
# Target coast is in bit 16.
ACTION_TARGET_PROVINCE_COAST = 16
# Third province (for support move, convoys) is in bits 25 to 31
# For a support move PAR S MAR-BUR, BUR is the target and MAR is the third.
# For a convoy ENG C LON-BRE, BRE is the target and LON is the third.
ACTION_THIRD_PROVINCE_START = 25
# Third province coast is in bit 24.
ACTION_THIRD_PROVINCE_COAST = 24
# Bits 0-31 contain the order representation. Bits 32-47 are always 0,
# bits 48-63 contain the action index into POSSIBLE_ACTIONS.
ACTION_INDEX_START = 48

# Order Type Codes
CONVOY_TO = 1
CONVOY = 2
HOLD = 3
MOVE_TO = 4
SUPPORT_HOLD = 5
SUPPORT_MOVE_TO = 6
DISBAND = 8
RETREAT_TO = 9
BUILD_ARMY = 10
BUILD_FLEET = 11
REMOVE = 12
WAIVE = 13

# Useful Constants.

# Actions in action_list.py
POSSIBLE_ACTIONS = action_list.POSSIBLE_ACTIONS

# Number of possible actions in action_list.py.
MAX_ACTION_INDEX = len(POSSIBLE_ACTIONS)

# Maximum number of orders for a single player.
MAX_ORDERS = 17

# The maximum number of legal actions for a single unit. This is approximate,
# but the board configurations required to produce it are extremely contrived.
MAX_UNIT_LEGAL_ACTIONS = 700

# The maximum number of legal actions for a single order. This is a conservative
# bound, given MAX_UNIT_LEGAL_ACTIONS.
MAX_LEGAL_ACTIONS = MAX_UNIT_LEGAL_ACTIONS * MAX_ORDERS

# Typing:
Order = int  # One of the Order Type Codes above
Action = int  # One of action_list.POSSIBLE_ACTIONS
ActionNoIndex = int  # A 32 bit version of the action, without the index.


def bits_between(number: int, start: int, end: int):
  """Returns bits between positions start and end from number."""
  return number % (1 << end) // (1 << start)


def actions_for_province(legal_actions: Sequence[Action],
                         province: utils.ProvinceID) -> Sequence[Action]:
  """Returns all actions in legal_actions with main unit in province."""
  actions = []
  for action in legal_actions:
    action_province = ordered_province(action)
    if action and action_province == province:
      actions.append(action)
  return actions


def construct_action(
    order: Order,
    ordering_province: utils.ProvinceWithFlag,
    target_province: utils.ProvinceWithFlag,
    third_province: utils.ProvinceWithFlag) -> ActionNoIndex:
  """Construct the action for this order, without the action index."""
  order_rep = 0
  order_rep |= order << ACTION_ORDER_START
  if ordering_province is not None:
    order_rep |= ordering_province[0] << ACTION_ORDERED_PROVINCE_START
    if order == BUILD_FLEET:
      order_rep |= ordering_province[1] << ACTION_ORDERED_PROVINCE_COAST
  if target_province is not None:
    order_rep |= target_province[0] << ACTION_TARGET_PROVINCE_START
    if order == MOVE_TO or order == RETREAT_TO:
      # For moves and retreats, we need to specify the coast.
      order_rep |= target_province[1] << ACTION_TARGET_PROVINCE_COAST
  if third_province is not None:
    order_rep |= third_province[0] << ACTION_THIRD_PROVINCE_START
  return order_rep


def action_breakdown(action: Union[Action, ActionNoIndex]) -> Tuple[
    int, utils.ProvinceWithFlag,
    utils.ProvinceWithFlag, utils.ProvinceWithFlag]:
  """Break down an action into its component parts.

  WARNING: The coast indicator bits returned by this function are not area_ids
  as returned by province_id and area.

  Args:
    action: 32bit or 64bit integer action

  Returns:
    - order: an integer between 1 and 13
    - p1: province_id a coast indicator bit
    - p2: province_id a coast indicator bit
    - p3: province_id a coast indicator bit
  """
  order = bits_between(action, ACTION_ORDER_START,
                       ACTION_ORDER_START+ACTION_ORDER_BITS)
  p1 = (bits_between(action, ACTION_ORDERED_PROVINCE_START,
                     ACTION_ORDERED_PROVINCE_START+ACTION_PROVINCE_BITS),
        bits_between(action,
                     ACTION_ORDERED_PROVINCE_COAST,
                     ACTION_ORDERED_PROVINCE_COAST+1)
       )
  p2 = (bits_between(action,
                     ACTION_TARGET_PROVINCE_START,
                     ACTION_TARGET_PROVINCE_START+ACTION_PROVINCE_BITS),
        bits_between(action,
                     ACTION_TARGET_PROVINCE_COAST,
                     ACTION_TARGET_PROVINCE_COAST+1)
       )
  p3 = (bits_between(action,
                     ACTION_THIRD_PROVINCE_START,
                     ACTION_THIRD_PROVINCE_START+ACTION_PROVINCE_BITS),
        bits_between(action,
                     ACTION_THIRD_PROVINCE_COAST,
                     ACTION_THIRD_PROVINCE_COAST+1)
       )
  return order, p1, p2, p3


def action_index(action: Union[Action, np.ndarray]) -> Union[int, np.ndarray]:
  """Returns the actions index among all possible unit actions."""
  return action >> ACTION_INDEX_START


def is_waive(action: Union[Action, ActionNoIndex]) -> bool:
  order = bits_between(action, ACTION_ORDER_START,
                       ACTION_ORDER_START+ACTION_ORDER_BITS)
  return order == WAIVE


def ordered_province(action: Union[Action, ActionNoIndex, np.ndarray]
                    ) -> Union[utils.ProvinceID, np.ndarray]:
  return bits_between(action, ACTION_ORDERED_PROVINCE_START,
                      ACTION_ORDERED_PROVINCE_START+ACTION_PROVINCE_BITS)


def shrink_actions(
    actions: Union[Action, Sequence[Action], np.ndarray]
) -> np.ndarray:
  """Retains the top and bottom byte pairs of actions.

  The "shrunk" action retains the top and bottom byte pairs containing contain
  the index, the order, and the ordered unit's area.

  Args:
    actions: action(s) in the format descrived at the top of this file.

  Returns:
    shrunk actions.
  """
  actions = np.asarray(actions)
  if actions.size == 0:
    return actions.astype(np.int32)
  return np.cast[np.int32](((actions >> 32) & ~0xffff) + (actions & 0xffff))


def find_action_with_area(actions: Sequence[Union[Action, ActionNoIndex]],
                          area: utils.AreaID) -> int:
  """The first action in the list for a unit in this area. 0 if None exists."""
  province = utils.province_id_and_area_index(area)[0]
  for a in actions:
    if ordered_province(a) == province:
      return a
  return 0

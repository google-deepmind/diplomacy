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

from typing import Optional, Union

import numpy as np

from diplomacy.environment import action_utils
from diplomacy.environment import observation_utils as utils
from diplomacy.environment import province_order


tag_to_id = province_order.province_name_to_id()
_province_id_to_tag = {v: k for k, v in tag_to_id.items()}


def area_string(area_tuple: utils.ProvinceWithFlag) -> str:
  return _province_id_to_tag[area_tuple[0]]


def area_string_with_coast_if_fleet(
    area_tuple: utils.ProvinceWithFlag,
    unit_type: Optional[utils.UnitType]):
  """String representation indicating coasts when fleet in bicoastal."""
  province_id, coast_num = area_tuple
  if (province_id < utils.SINGLE_COASTED_PROVINCES or
      unit_type == utils.UnitType.ARMY):
    return area_string(area_tuple)
  elif unit_type == utils.UnitType.FLEET:
    # Fleet in a bicoastal province
    province_tag = _province_id_to_tag[province_id]
    return province_tag + ('NC' if coast_num == 0 else 'SC')
  elif unit_type is None:
    # Unit type unknown to caller
    province_tag = _province_id_to_tag[province_id]
    return province_tag + ('maybe_NC' if coast_num == 0 else 'SC')
  else:
    raise ValueError('Invalid unit type')


def action_string(
    action: Union[action_utils.Action, action_utils.ActionNoIndex],
    board: Optional[np.ndarray],
) -> str:
  """Returns a human readable action string.

  Args:
    action: The action to write down
    board: (optional) board, as part of the observation. This is used to know
     whether units are fleets or not, for coast annotations

  Returns:
    Action in an abbreviated human notation.
  """
  order, p1, p2, p3 = action_utils.action_breakdown(action)

  unit_string = area_string(p1)
  if board is None:
    unit_type = None
  else:
    unit_type = utils.unit_type(p1[0], board)

  if order == action_utils.HOLD:
    return f'{unit_string} H'
  elif order == action_utils.CONVOY:
    return f'{unit_string} C {area_string(p3)} - {area_string(p2)}'
  elif order == action_utils.CONVOY_TO:
    return f'{unit_string} - {area_string(p2)} VC'
  elif order == action_utils.MOVE_TO:
    return f'{unit_string} - {area_string_with_coast_if_fleet(p2, unit_type)}'
  elif order == action_utils.SUPPORT_HOLD:
    return f'{unit_string} SH {area_string(p2)}'
  elif order == action_utils.SUPPORT_MOVE_TO:
    return f'{unit_string} S {area_string(p3)} - {area_string(p2)}'
  elif order == action_utils.RETREAT_TO:
    return f'{unit_string} - {area_string_with_coast_if_fleet(p2, unit_type)}'
  elif order == action_utils.DISBAND:
    return f'{unit_string} D'
  elif order == action_utils.BUILD_ARMY:
    return 'B A ' + area_string(p1)
  elif order == action_utils.BUILD_FLEET:
    return 'B F ' + area_string_with_coast_if_fleet(p1, utils.UnitType.FLEET)
  elif order == action_utils.REMOVE:
    return 'R ' + area_string(p1)
  elif order == action_utils.WAIVE:
    return 'W'
  else:
    raise ValueError('Unrecognised order %s ' % order)

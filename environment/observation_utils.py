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

import collections
import enum
from typing import Optional, Sequence, Tuple

import numpy as np


# Indices of bits representing global state.
NUM_POWERS = 7
# Max number of units that might need an order in total, across all players.
MAX_ACTIONS = 34

# Bits representing interesting things in per-province state vectors
OBSERVATION_UNIT_ARMY = 0
OBSERVATION_UNIT_FLEET = 1
OBSERVATION_UNIT_ABSENT = 2
OBSERVATION_UNIT_POWER_START = 3
OBSERVATION_BUILDABLE = 11
OBSERVATION_REMOVABLE = 12
OBSERVATION_DISLODGED_ARMY = 13
OBSERVATION_DISLODGED_FLEET = 14
OBSERVATION_DISLODGED_START = 16
OBSERVATION_SC_POWER_START = 27
PROVINCE_VECTOR_LENGTH = 35

# Areas information. The observation has NUM_AREAS area vectors:
# - First, a vector for each of the SINGLE_COASTED_PROVINCES provinces.
# - Then, three vectors for each of the BICOASTAL_PROVINCES. The first one is
#   the land area, and the other two are the coasts.
SINGLE_COASTED_PROVINCES = 72
BICOASTAL_PROVINCES = 3
NUM_AREAS = SINGLE_COASTED_PROVINCES + 3 * BICOASTAL_PROVINCES
NUM_PROVINCES = SINGLE_COASTED_PROVINCES + BICOASTAL_PROVINCES
NUM_SUPPLY_CENTERS = 34

OBSERVATION_BOARD_SHAPE = [NUM_AREAS, PROVINCE_VECTOR_LENGTH]

# Agents may want a list of areas that they can take actions in. This
# doesn't make sense in the builds phase, so we use the value -2 to indicate
# that, and will use lists like [BUILD_PHASE_AREA]*num_builds as 'area lists'
BUILD_PHASE_AREA_FLAG = -2
INVALID_AREA_FLAG = -1

# Main areas_ids of bicoastal provinces
BICOASTAL_PROVINCES_MAIN_AREAS = [72, 75, 78]

ProvinceID = int  # in [0, 1, ...74]
AreaID = int  # in [0, 1, ...80]
# AreaIndex is 0 for the "land" area of bicoastal provinces and for any province
# that has either 0 or 1 coasts. AreaIndex is 1 for the first coast and 2 for
# the second coast of bicoastal provinces.
AreaIndex = int  # in [0, 1, 2]
# CoastFlag is 0 for the first and 1 for the second coast of bicoastal
# provinces. CoastFlag is 0 for all provinces that have 0 or 1 coast.
CoastFlag = int  # in [0, 1]
ProvinceWithFlag = Tuple[ProvinceID, CoastFlag]


class Season(enum.Enum):
  """Diplomacy season."""
  SPRING_MOVES = 0
  SPRING_RETREATS = 1
  AUTUMN_MOVES = 2
  AUTUMN_RETREATS = 3
  BUILDS = 4

  def is_moves(self):
    return self == Season.SPRING_MOVES or self == Season.AUTUMN_MOVES

  def is_retreats(self):
    return self == Season.SPRING_RETREATS or self == Season.AUTUMN_RETREATS

  def is_builds(self):
    return self == Season.BUILDS

NUM_SEASONS = len(Season)


class UnitType(enum.Enum):
  ARMY = 0
  FLEET = 1

Observation = collections.namedtuple(
    'Observation', ['season', 'board', 'build_numbers', 'last_actions'])


class ProvinceType(enum.Enum):
  LAND = 0
  SEA = 1
  COASTAL = 2
  BICOASTAL = 3


def province_type_from_id(province_id: ProvinceID) -> ProvinceType:
  """Returns the ProvinceType for the province."""
  if province_id < 14:
    return ProvinceType.LAND
  elif province_id < 33:
    return ProvinceType.SEA
  elif province_id < 72:
    return ProvinceType.COASTAL
  elif province_id < 75:
    return ProvinceType.BICOASTAL
  else:
    raise ValueError('Invalid ProvinceID (too large)')


def province_id_and_area_index(area: AreaID) -> Tuple[ProvinceID, AreaIndex]:
  """Returns the province_id and the area index within the province.

  Args:
    area: the ID of the area in the observation vector, an integer from 0 to 80

  Returns:
    province_id: This is between 0 and
      SINGLE_COASTED_PROVINCES + BICOASTAL_PROVINCES - 1, and corresponds to the
      representation of this area used in orders.
    area_index: this is 0 for the main area of a province, and 1 or 2 for a
      coast in a bicoastal province.
  """
  if area < SINGLE_COASTED_PROVINCES:
    return area, 0
  province_id = (
      SINGLE_COASTED_PROVINCES + (area - SINGLE_COASTED_PROVINCES) // 3)
  area_index = (area - SINGLE_COASTED_PROVINCES) % 3
  return province_id, area_index


_prov_and_area_id_to_area = {province_id_and_area_index(area): area
                             for area in range(NUM_AREAS)}


def area_from_province_id_and_area_index(province_id: ProvinceID,
                                         area_index: AreaIndex) -> AreaID:
  """The inverse of province_id_and_area_index.

  Args:
    province_id: This is between 0 and
      SINGLE_COASTED_PROVINCES + BICOASTAL_PROVINCES - 1, and corresponds to the
      representation of this area used in orders.
    area_index: this is 0 for the main area of a province, and 1 or 2 for a
      coast in a bicoastal province.

  Returns:
    area: the id of the area in the observation vector

  Raises:
    KeyError: If the province_id and area_index are invalid
  """
  return _prov_and_area_id_to_area[(province_id, area_index)]


def area_index_for_fleet(
    province_tuple: ProvinceWithFlag) -> AreaIndex:
  if province_type_from_id(province_tuple[0]) == ProvinceType.BICOASTAL:
    return province_tuple[1] + 1
  else:
    return 0


def obs_index_start_and_num_areas(
    province_id: ProvinceID) -> Tuple[AreaID, int]:
  """Returns the area_id of the province's main area, and the number of areas.

  Args:
    province_id: the id of the province.
  """
  if province_id < SINGLE_COASTED_PROVINCES:
    return province_id, 1
  area_start = (
      SINGLE_COASTED_PROVINCES + (province_id - SINGLE_COASTED_PROVINCES) * 3)
  return area_start, 3


def moves_phase_areas(country_index: int, board_state: np.ndarray,
                      retreats: bool) -> Sequence[AreaID]:
  """Returns the areas with country_index's units active for this phase."""
  offset = (OBSERVATION_DISLODGED_START if retreats else
            OBSERVATION_UNIT_POWER_START)
  our_areas = np.where(board_state[:, country_index + offset])[0]
  filtered_areas = []
  provinces = set()
  for area in our_areas:
    province_id, area_index = province_id_and_area_index(area)
    if retreats:
      u_type = dislodged_unit_type(province_id, board_state)
    else:
      u_type = unit_type(province_id, board_state)
    # This area is valid, unless the unit is a fleet, this is the first area of
    # its province, and the province is bicoastal.
    if (u_type == UnitType.FLEET and
        area_index == 0 and
        obs_index_start_and_num_areas(province_id)[1] > 1):
      continue
    filtered_areas.append(area)
    if province_id in provinces:
      raise ValueError('Duplicate province in move phase areas')
    provinces.add(province_id)
  return sorted(list(filtered_areas))


def order_relevant_areas(observation: Observation, player: int,
                         topological_index=None) -> Sequence[AreaID]:
  """Areas with moves sorted according to topological_index."""
  season = observation.season
  if season.is_moves():
    areas = moves_phase_areas(player, observation.board, False)
  elif season.is_retreats():
    areas = moves_phase_areas(player, observation.board, True)
  else:
    areas = [BUILD_PHASE_AREA_FLAG] * abs(
        observation.build_numbers[player])
    return areas

  provinces_to_areas = dict()
  for area in areas:
    province, _ = province_id_and_area_index(area)
    if (province not in provinces_to_areas or
        area > provinces_to_areas[province]):  # This selects coasts over land
      provinces_to_areas[province] = area

  areas_without_repeats = list(provinces_to_areas.values())
  if topological_index:
    areas_without_repeats.sort(key=topological_index.index)

  return areas_without_repeats


def unit_type(province_id: ProvinceID,
              board_state: np.ndarray) -> Optional[UnitType]:
  """Returns the unit type in the province."""
  main_area, _ = obs_index_start_and_num_areas(province_id)
  return unit_type_from_area(main_area, board_state)


def unit_type_from_area(area_id: AreaID,
                        board_state: np.ndarray) -> Optional[UnitType]:
  if board_state[area_id, OBSERVATION_UNIT_ARMY] > 0:
    return UnitType(UnitType.ARMY)
  elif board_state[area_id, OBSERVATION_UNIT_FLEET] > 0:
    return UnitType(UnitType.FLEET)
  return None


def dislodged_unit_type(province_id: ProvinceID,
                        board_state: np.ndarray) -> Optional[UnitType]:
  """Returns the type of any dislodged unit in the province."""
  main_area, _ = obs_index_start_and_num_areas(province_id)
  return dislodged_unit_type_from_area(main_area, board_state)


def dislodged_unit_type_from_area(
    area_id: AreaID, board_state: np.ndarray) -> Optional[UnitType]:
  """Returns the type of any dislodged unit in the province."""
  if board_state[area_id, OBSERVATION_DISLODGED_ARMY] > 0:
    return UnitType(UnitType.ARMY)
  elif board_state[area_id, OBSERVATION_DISLODGED_FLEET] > 0:
    return UnitType(UnitType.FLEET)
  return None


def unit_power(province_id: ProvinceID,
               board_state: np.ndarray) -> Optional[int]:
  """Returns which power controls the unit province (None if no unit there)."""
  main_area, _ = obs_index_start_and_num_areas(province_id)
  return unit_power_from_area(main_area, board_state)


def unit_power_from_area(area_id: AreaID,
                         board_state: np.ndarray) -> Optional[int]:
  if unit_type_from_area(area_id, board_state) is None:
    return None

  for power_id in range(NUM_POWERS):
    if board_state[area_id, OBSERVATION_UNIT_POWER_START + power_id]:
      return power_id
  raise ValueError('Expected a unit there, but none of the powers indicated')


def dislodged_unit_power(province_id: ProvinceID,
                         board_state: np.ndarray) -> Optional[int]:
  """Returns which power controls the unit province (None if no unit there)."""
  main_area, _ = obs_index_start_and_num_areas(province_id)
  return dislodged_unit_power_from_area(main_area, board_state)


def dislodged_unit_power_from_area(area_id: AreaID,
                                   board_state: np.ndarray) -> Optional[int]:
  if unit_type_from_area(area_id, board_state) is None:
    return None

  for power_id in range(NUM_POWERS):
    if board_state[area_id, OBSERVATION_DISLODGED_START + power_id]:
      return power_id
  raise ValueError('Expected a unit there, but none of the powers indicated')


def build_areas(country_index: int,
                board_state: np.ndarray) -> Sequence[AreaID]:
  """Returns all areas where it is legal for a power to build.

  Args:
    country_index: The power to get provinces for.
    board_state: Board from observation.
  """
  return np.where(
      np.logical_and(
          board_state[:, country_index + OBSERVATION_SC_POWER_START] > 0,
          board_state[:, OBSERVATION_BUILDABLE] > 0))[0]


def build_provinces(country_index: int,
                    board_state: np.ndarray) -> Sequence[ProvinceID]:
  """Returns all provinces where it is legal for a power to build.

  This returns province IDs, not area numbers.

  Args:
    country_index: The power to get provinces for.
    board_state: Board from observation.
  """
  buildable_provinces = []
  for a in build_areas(country_index, board_state):
    province_id, area_index = province_id_and_area_index(a)
    if area_index != 0:
      # We get only the main province.
      continue
    buildable_provinces.append(province_id)
  return buildable_provinces


def sc_provinces(country_index: int,
                 board_state: np.ndarray) -> Sequence[ProvinceID]:
  """Returns all supply centres the power owns.

  This returns province IDs, not area IDs.

  Args:
    country_index: The power to get provinces for.
    board_state: Board from observation.
  """
  sc_areas = np.where(
      board_state[:, country_index + OBSERVATION_SC_POWER_START] > 0)[0]
  provinces = []
  for a in sc_areas:
    province_id, area_index = province_id_and_area_index(a)
    if area_index != 0:
      # We get only the main province.
      continue
    provinces.append(province_id)
  return provinces


def removable_areas(country_index: int,
                    board_state: np.ndarray) -> Sequence[AreaID]:
  """Get all areas where it is legal for a power to remove."""
  return np.where(
      np.logical_and(
          board_state[:, country_index + OBSERVATION_UNIT_POWER_START] > 0,
          board_state[:, OBSERVATION_REMOVABLE] > 0))[0]


def removable_provinces(country_index: int,
                        board_state: np.ndarray) -> Sequence[ProvinceID]:
  """Get all provinces where it is legal for a power to remove."""
  remove_provinces = []
  for a in removable_areas(country_index, board_state):
    province_id, area_index = province_id_and_area_index(a)
    if area_index != 0:
      # We get only the main province.
      continue
    remove_provinces.append(province_id)
  return remove_provinces


def area_id_for_unit_in_province_id(province_id: ProvinceID,
                                    board_state: np.ndarray) -> AreaID:
  """AreaID from [0..80] of the unit in board_state."""
  if unit_type(province_id, board_state) is None:
    raise ValueError('No unit in province {}'.format(province_id))

  if (obs_index_start_and_num_areas(province_id)[1] == 3 and
      unit_type(province_id, board_state) == UnitType.FLEET):
    first_coast = area_from_province_id_and_area_index(province_id, 1)
    if unit_type_from_area(first_coast, board_state) is not None:
      return area_from_province_id_and_area_index(province_id, 1)
    else:
      return area_from_province_id_and_area_index(province_id, 2)
  else:
    return area_from_province_id_and_area_index(province_id, 0)


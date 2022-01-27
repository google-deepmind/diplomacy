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

"""Contains basic facts about a Diplomacy map defined in MDF format.

We need two map descriptions in MDF format, a "standard map" that only considers
provinces as defined in Diplomacy, and a bi-coastal map that considers each
coast separately, including those that are belong to the same Diplomacy province
(e.g. Spain north and south coasts).
"""

import enum
from typing import Dict, Sequence
import numpy as np

from diplomacy.environment import observation_utils as utils


class MapMDF(enum.Enum):
  STANDARD_MAP = 0
  BICOASTAL_MAP = 1


def get_mdf_content(map_mdf: MapMDF = MapMDF.STANDARD_MAP) -> str:
  if map_mdf == MapMDF.STANDARD_MAP:
    return _STANDARD_MAP_MDF_CONTENT
  elif map_mdf == MapMDF.BICOASTAL_MAP:
    return _BICOASTAL_MAP_MDF_CONTENT
  raise ValueError(f'Unknown map_mdf: {map_mdf}')


def _province_tag(l: str) -> str:
  words = str(l).split(' ')
  for w in words:
    if w not in ['(', ')']:
      return w
  raise ValueError('No province found for line {}'.format(l))


def province_name_to_id(
    map_mdf: MapMDF = MapMDF.STANDARD_MAP
) -> Dict[str, utils.ProvinceID]:
  """Gets dictionary of province name to order in observation."""
  return _tag_to_id(get_mdf_content(map_mdf))


def province_id_to_home_sc_power() -> Dict[utils.ProvinceID, int]:
  """Which power is this a home sc for?"""
  content = get_mdf_content(MapMDF.STANDARD_MAP)
  home_sc_line = content.splitlines()[2]
  tag_to_id = _tag_to_id(get_mdf_content(MapMDF.STANDARD_MAP))

  # Assume powers are ordered correctly
  id_to_power = {}
  power = -1
  words = str(home_sc_line).split(' ')
  for w in words:
    if w in ['(', ')']:
      pass
    elif w in tag_to_id:  # Is a province
      id_to_power[tag_to_id[w]] = power
    else:  # Must be a power tag
      power += 1
  return id_to_power


def _tag_to_id(mdf_content: str) -> Dict[str, int]:
  tag_to_id = dict()
  tags_found = 0
  lines = mdf_content.splitlines()
  for l in lines[4:-1]:
    tag_to_id[_province_tag(l)] = tags_found  # pylint: disable=protected-access
    tags_found += 1
  return tag_to_id


def build_adjacency(mdf_content: str) -> np.ndarray:
  """Builds adjacency matrix from MDF content.

  Args:
    mdf_content: content of an mdf map file.
  Returns:
    a num_provinces-by-num_provinces adjacency matrix. Provinces are adjacent if
    there is a path for either an army or a fleet to move between them.
    Note that a provinces with multiple coasts (e.g. Spain in the STANDARD_MAP)
    are considered adjacent to all provinces that are reachable from any of its
    coasts.
  """
  tag_to_id = _tag_to_id(mdf_content)
  num_provinces = np.max(list(tag_to_id.values())) + 1
  adjacency = np.zeros((num_provinces, num_provinces), dtype=np.float32)
  lines = mdf_content.splitlines()
  for edge_string in lines[4:-1]:
    provinces = [w for w in edge_string.split(' ') if w not in ('(', ')', '')]
    sender_province = provinces[0]
    if len(sender_province) > 3:
      land_province = sender_province[:3]
      adjacency[tag_to_id[sender_province], tag_to_id[land_province]] = 1.0
      adjacency[tag_to_id[land_province], tag_to_id[sender_province]] = 1.0
    for receiver_province in provinces[1:]:
      if receiver_province in ('AMY', 'FLT'):
        continue
      adjacency[tag_to_id[sender_province],
                tag_to_id[receiver_province]] = 1.0
  return adjacency


def topological_index(
    mdf_content: str,
    topological_order: Sequence[str]
) -> Sequence[utils.ProvinceID]:
  tag_to_id = _tag_to_id(mdf_content)
  return [tag_to_id[province] for province in topological_order]


def fleet_adjacency_map() -> Dict[utils.AreaID, Sequence[utils.AreaID]]:
  """Builds a mapping for valid fleet movements between areas."""
  mdf_content = get_mdf_content(MapMDF.BICOASTAL_MAP)
  tag_to_area_id = _tag_to_id(mdf_content)
  lines = mdf_content.splitlines()

  fleet_adjacency = {}
  for edge_string in lines[4:-1]:
    provinces = [w for w in edge_string.split(' ') if w not in ('(', ')', '')]
    start_province = tag_to_area_id[provinces[0]]
    fleet_adjacency[start_province] = []
    flt_tag_found = False
    for province in provinces[1:]:
      if flt_tag_found:
        fleet_adjacency[start_province].append(tag_to_area_id[province])
      elif province == 'FLT':
        flt_tag_found = True

  return fleet_adjacency


_STANDARD_MAP_MDF_CONTENT = """MDF
( AUS ENG FRA GER ITA RUS TUR )
( ( ( AUS VIE BUD TRI ) ( ENG LON EDI LVP ) ( FRA PAR MAR BRE ) ( GER BER MUN KIE ) ( ITA ROM VEN NAP ) ( RUS MOS SEV WAR STP ) ( TUR ANK CON SMY ) ( UNO SER BEL DEN GRE HOL NWY POR RUM SWE TUN BUL SPA ) )
( BOH BUR GAL RUH SIL TYR UKR ADR AEG BAL BAR BLA EAS ECH GOB GOL HEL ION IRI MAO NAO NTH NWG SKA TYS WES ALB APU ARM CLY FIN GAS LVN NAF PIC PIE PRU SYR TUS WAL YOR ) )
( ( BOH ( AMY GAL SIL TYR MUN VIE ) ) ) )
( BUR ( AMY RUH MUN PAR GAS PIC BEL MAR ) ) ) )
( GAL ( AMY BOH SIL UKR BUD VIE WAR RUM ) ) ) )
( RUH ( AMY BUR MUN BEL HOL KIE ) ) ) )
( SIL ( AMY BOH GAL MUN WAR PRU BER ) ) ) )
( TYR ( AMY BOH MUN VIE PIE TRI VEN ) ) ) )
( UKR ( AMY GAL MOS WAR RUM SEV ) ) ) )
( BUD ( AMY GAL SER VIE RUM TRI ) ) ) )
( MOS ( AMY UKR WAR LVN SEV STP ) ) ) )
( MUN ( AMY BOH BUR RUH SIL TYR BER KIE ) ) ) )
( PAR ( AMY BUR GAS PIC BRE ) ) ) )
( SER ( AMY BUD ALB GRE RUM TRI BUL ) ) ) )
( VIE ( AMY BOH GAL TYR BUD TRI ) ) ) )
( WAR ( AMY GAL SIL UKR MOS LVN PRU ) ) ) )
( ADR ( FLT ION ALB APU TRI VEN ) ) ) )
( AEG ( FLT EAS ION CON GRE SMY BUL ) ) ) )
( BAL ( FLT GOB LVN PRU BER DEN KIE SWE ) ) ) )
( BAR ( FLT NWG NWY STP ) ) ) )
( BLA ( FLT ARM ANK CON RUM SEV BUL ) ) ) )
( EAS ( FLT AEG ION SYR SMY ) ) ) )
( ECH ( FLT IRI MAO NTH PIC WAL BEL BRE LON ) ) ) )
( GOB ( FLT BAL FIN LVN SWE STP ) ) ) )
( GOL ( FLT TYS WES PIE TUS MAR SPA ) ) ) )
( HEL ( FLT NTH DEN HOL KIE ) ) ) )
( ION ( FLT ADR AEG EAS TYS ALB APU GRE NAP TUN ) ) ) )
( IRI ( FLT ECH MAO NAO WAL LVP ) ) ) )
( MAO ( FLT ECH IRI NAO WES GAS NAF BRE POR SPA SPA ) ) ) )
( NAO ( FLT IRI MAO NWG CLY LVP ) ) ) )
( NTH ( FLT ECH HEL NWG SKA YOR BEL DEN EDI HOL LON NWY ) ) ) )
( NWG ( FLT BAR NAO NTH CLY EDI NWY ) ) ) )
( SKA ( FLT NTH DEN NWY SWE ) ) ) )
( TYS ( FLT GOL ION WES TUS NAP ROM TUN ) ) ) )
( WES ( FLT GOL MAO TYS NAF TUN SPA ) ) ) )
( ALB ( AMY SER GRE TRI ) ( FLT ADR ION GRE TRI ) ) )
( APU ( AMY NAP ROM VEN ) ( FLT ADR ION NAP VEN ) ) )
( ARM ( AMY SYR ANK SEV SMY ) ( FLT BLA ANK SEV ) ) )
( CLY ( AMY EDI LVP ) ( FLT NAO NWG EDI LVP ) ) )
( FIN ( AMY NWY SWE STP ) ( FLT GOB SWE STP ) ) )
( GAS ( AMY BUR PAR BRE MAR SPA ) ( FLT MAO BRE SPA ) ) )
( LVN ( AMY MOS WAR PRU STP ) ( FLT BAL GOB PRU STP ) ) )
( NAF ( AMY TUN ) ( FLT MAO WES TUN ) ) )
( PIC ( AMY BUR PAR BEL BRE ) ( FLT ECH BEL BRE ) ) )
( PIE ( AMY TYR TUS MAR VEN ) ( FLT GOL TUS MAR ) ) )
( PRU ( AMY SIL WAR LVN BER ) ( FLT BAL LVN BER ) ) )
( SYR ( AMY ARM SMY ) ( FLT EAS SMY ) ) )
( TUS ( AMY PIE ROM VEN ) ( FLT GOL TYS PIE ROM ) ) )
( WAL ( AMY YOR LON LVP ) ( FLT ECH IRI LON LVP ) ) )
( YOR ( AMY WAL EDI LON LVP ) ( FLT NTH EDI LON ) ) )
( ANK ( AMY ARM CON SMY ) ( FLT BLA ARM CON ) ) )
( BEL ( AMY BUR RUH PIC HOL ) ( FLT ECH NTH PIC HOL ) ) )
( BER ( AMY SIL MUN PRU KIE ) ( FLT BAL PRU KIE ) ) )
( BRE ( AMY PAR GAS PIC ) ( FLT ECH MAO GAS PIC ) ) )
( CON ( AMY ANK SMY BUL ) ( FLT AEG BLA ANK SMY BUL BUL ) ) )
( DEN ( AMY KIE SWE ) ( FLT BAL HEL NTH SKA KIE SWE ) ) )
( EDI ( AMY CLY YOR LVP ) ( FLT NTH NWG CLY YOR ) ) )
( GRE ( AMY SER ALB BUL ) ( FLT AEG ION ALB BUL ) ) )
( HOL ( AMY RUH BEL KIE ) ( FLT HEL NTH BEL KIE ) ) )
( KIE ( AMY RUH MUN BER DEN HOL ) ( FLT BAL HEL BER DEN HOL ) ) )
( LON ( AMY WAL YOR ) ( FLT ECH NTH WAL YOR ) ) )
( LVP ( AMY CLY WAL YOR EDI ) ( FLT IRI NAO CLY WAL ) ) )
( MAR ( AMY BUR GAS PIE SPA ) ( FLT GOL PIE SPA ) ) )
( NAP ( AMY APU ROM ) ( FLT ION TYS APU ROM ) ) )
( NWY ( AMY FIN SWE STP ) ( FLT BAR NTH NWG SKA SWE STP ) ) )
( POR ( AMY SPA ) ( FLT MAO SPA SPA ) ) )
( ROM ( AMY APU TUS NAP VEN ) ( FLT TYS TUS NAP ) ) )
( RUM ( AMY GAL UKR BUD SER SEV BUL ) ( FLT BLA SEV BUL ) ) )
( SEV ( AMY UKR MOS ARM RUM ) ( FLT BLA ARM RUM ) ) )
( SMY ( AMY ARM SYR ANK CON ) ( FLT AEG EAS SYR CON ) ) )
( SWE ( AMY FIN DEN NWY ) ( FLT BAL GOB SKA FIN DEN NWY ) ) )
( TRI ( AMY TYR BUD SER VIE ALB VEN ) ( FLT ADR ALB VEN ) ) )
( TUN ( AMY NAF ) ( FLT ION TYS WES NAF ) ) )
( VEN ( AMY TYR APU PIE TUS ROM TRI ) ( FLT ADR APU TRI ) ) )
( BUL ( AMY SER CON GRE RUM ) ( FLT BLA CON RUM ) ( FLT AEG CON GRE ) )
( SPA ( AMY GAS MAR POR ) ( FLT MAO GAS POR ) ( FLT GOL MAO WES MAR POR ) )
( STP ( AMY MOS FIN LVN NWY ) ( FLT BAR NWY ) ( FLT GOB FIN LVN ) )
)
"""

_BICOASTAL_MAP_MDF_CONTENT = """MDF
( AUS ENG FRA GER ITA RUS TUR )
( ( ( AUS VIE BUD TRI ) ( ENG LON EDI LVP ) ( FRA PAR MAR BRE ) ( GER BER MUN KIE ) ( ITA ROM VEN NAP ) ( RUS MOS SEV WAR STP ) ( TUR ANK CON SMY ) ( UNO SER BEL DEN GRE HOL NWY POR RUM SWE TUN BUL SPA ) )
( BOH BUR GAL RUH SIL TYR UKR ADR AEG BAL BAR BLA EAS ECH GOB GOL HEL ION IRI MAO NAO NTH NWG SKA TYS WES ALB APU ARM CLY FIN GAS LVN NAF PIC PIE PRU SYR TUS WAL YOR ) )
( ( BOH ( AMY GAL SIL TYR MUN VIE ) ) ) )
( BUR ( AMY RUH MUN PAR GAS PIC BEL MAR ) ) ) )
( GAL ( AMY BOH SIL UKR BUD VIE WAR RUM ) ) ) )
( RUH ( AMY BUR MUN BEL HOL KIE ) ) ) )
( SIL ( AMY BOH GAL MUN WAR PRU BER ) ) ) )
( TYR ( AMY BOH MUN VIE PIE TRI VEN ) ) ) )
( UKR ( AMY GAL MOS WAR RUM SEV ) ) ) )
( BUD ( AMY GAL SER VIE RUM TRI ) ) ) )
( MOS ( AMY UKR WAR LVN SEV STP ) ) ) )
( MUN ( AMY BOH BUR RUH SIL TYR BER KIE ) ) ) )
( PAR ( AMY BUR GAS PIC BRE ) ) ) )
( SER ( AMY BUD ALB GRE RUM TRI BUL ) ) ) )
( VIE ( AMY BOH GAL TYR BUD TRI ) ) ) )
( WAR ( AMY GAL SIL UKR MOS LVN PRU ) ) ) )
( ADR ( FLT ION ALB APU TRI VEN ) ) ) )
( AEG ( FLT EAS ION CON GRE SMY BUL/SC ) ) ) )
( BAL ( FLT GOB LVN PRU BER DEN KIE SWE ) ) ) )
( BAR ( FLT NWG NWY STP/NC ) ) ) )
( BLA ( FLT ARM ANK CON RUM SEV BUL/EC ) ) ) )
( EAS ( FLT AEG ION SYR SMY ) ) ) )
( ECH ( FLT IRI MAO NTH PIC WAL BEL BRE LON ) ) ) )
( GOB ( FLT BAL FIN LVN SWE STP/SC ) ) ) )
( GOL ( FLT TYS WES PIE TUS MAR SPA/SC ) ) ) )
( HEL ( FLT NTH DEN HOL KIE ) ) ) )
( ION ( FLT ADR AEG EAS TYS ALB APU GRE NAP TUN ) ) ) )
( IRI ( FLT ECH MAO NAO WAL LVP ) ) ) )
( MAO ( FLT ECH IRI NAO WES GAS NAF BRE POR SPA/NC SPA/SC ) ) ) )
( NAO ( FLT IRI MAO NWG CLY LVP ) ) ) )
( NTH ( FLT ECH HEL NWG SKA YOR BEL DEN EDI HOL LON NWY ) ) ) )
( NWG ( FLT BAR NAO NTH CLY EDI NWY ) ) ) )
( SKA ( FLT NTH DEN NWY SWE ) ) ) )
( TYS ( FLT GOL ION WES TUS NAP ROM TUN ) ) ) )
( WES ( FLT GOL MAO TYS NAF TUN SPA/SC ) ) ) )
( ALB ( AMY SER GRE TRI ) ( FLT ADR ION GRE TRI ) ) )
( APU ( AMY NAP ROM VEN ) ( FLT ADR ION NAP VEN ) ) )
( ARM ( AMY SYR ANK SEV SMY ) ( FLT BLA ANK SEV ) ) )
( CLY ( AMY EDI LVP ) ( FLT NAO NWG EDI LVP ) ) )
( FIN ( AMY NWY SWE STP ) ( FLT GOB SWE STP/SC ) ) )
( GAS ( AMY BUR PAR BRE MAR SPA ) ( FLT MAO BRE SPA/NC ) ) )
( LVN ( AMY MOS WAR PRU STP ) ( FLT BAL GOB PRU STP/SC ) ) )
( NAF ( AMY TUN ) ( FLT MAO WES TUN ) ) )
( PIC ( AMY BUR PAR BEL BRE ) ( FLT ECH BEL BRE ) ) )
( PIE ( AMY TYR TUS MAR VEN ) ( FLT GOL TUS MAR ) ) )
( PRU ( AMY SIL WAR LVN BER ) ( FLT BAL LVN BER ) ) )
( SYR ( AMY ARM SMY ) ( FLT EAS SMY ) ) )
( TUS ( AMY PIE ROM VEN ) ( FLT GOL TYS PIE ROM ) ) )
( WAL ( AMY YOR LON LVP ) ( FLT ECH IRI LON LVP ) ) )
( YOR ( AMY WAL EDI LON LVP ) ( FLT NTH EDI LON ) ) )
( ANK ( AMY ARM CON SMY ) ( FLT BLA ARM CON ) ) )
( BEL ( AMY BUR RUH PIC HOL ) ( FLT ECH NTH PIC HOL ) ) )
( BER ( AMY SIL MUN PRU KIE ) ( FLT BAL PRU KIE ) ) )
( BRE ( AMY PAR GAS PIC ) ( FLT ECH MAO GAS PIC ) ) )
( CON ( AMY ANK SMY BUL ) ( FLT AEG BLA ANK SMY BUL/SC BUL/EC ) ) )
( DEN ( AMY KIE SWE ) ( FLT BAL HEL NTH SKA KIE SWE ) ) )
( EDI ( AMY CLY YOR LVP ) ( FLT NTH NWG CLY YOR ) ) )
( GRE ( AMY SER ALB BUL ) ( FLT AEG ION ALB BUL/SC ) ) )
( HOL ( AMY RUH BEL KIE ) ( FLT HEL NTH BEL KIE ) ) )
( KIE ( AMY RUH MUN BER DEN HOL ) ( FLT BAL HEL BER DEN HOL ) ) )
( LON ( AMY WAL YOR ) ( FLT ECH NTH WAL YOR ) ) )
( LVP ( AMY CLY WAL YOR EDI ) ( FLT IRI NAO CLY WAL ) ) )
( MAR ( AMY BUR GAS PIE SPA ) ( FLT GOL PIE SPA/SC ) ) )
( NAP ( AMY APU ROM ) ( FLT ION TYS APU ROM ) ) )
( NWY ( AMY FIN SWE STP ) ( FLT BAR NTH NWG SKA SWE STP/NC ) ) )
( POR ( AMY SPA ) ( FLT MAO SPA/NC SPA/SC ) ) )
( ROM ( AMY APU TUS NAP VEN ) ( FLT TYS TUS NAP ) ) )
( RUM ( AMY GAL UKR BUD SER SEV BUL ) ( FLT BLA SEV BUL/EC ) ) )
( SEV ( AMY UKR MOS ARM RUM ) ( FLT BLA ARM RUM ) ) )
( SMY ( AMY ARM SYR ANK CON ) ( FLT AEG EAS SYR CON ) ) )
( SWE ( AMY FIN DEN NWY ) ( FLT BAL GOB SKA FIN DEN NWY ) ) )
( TRI ( AMY TYR BUD SER VIE ALB VEN ) ( FLT ADR ALB VEN ) ) )
( TUN ( AMY NAF ) ( FLT ION TYS WES NAF ) ) )
( VEN ( AMY TYR APU PIE TUS ROM TRI ) ( FLT ADR APU TRI ) ) )
( BUL ( AMY SER CON GRE RUM ) )
( BUL/EC ( FLT BLA CON RUM ) )
( BUL/SC ( FLT AEG CON GRE ) )
( SPA ( AMY GAS MAR POR ) )
( SPA/NC ( FLT MAO GAS POR ) )
( SPA/SC ( FLT GOL MAO WES MAR POR ) )
( STP ( AMY MOS FIN LVN NWY ) )
( STP/NC ( FLT BAR NWY ) )
( STP/SC ( FLT GOB FIN LVN ) )
)
"""

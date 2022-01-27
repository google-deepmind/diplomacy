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

"""Tests the action conversions defined in mila_actions.py."""
import collections

from absl.testing import absltest
from diplomacy.environment import action_list
from diplomacy.environment import action_utils
from diplomacy.environment import human_readable_actions
from diplomacy.environment import mila_actions


class MilaActionsTest(absltest.TestCase):

  def test_inversion_dm_actions(self):
    """Tests converting a DM to MILA to DM action recovers original action."""
    for original_action in action_list.POSSIBLE_ACTIONS:
      possible_mila_actions = mila_actions.action_to_mila_actions(
          original_action)
      for mila_action in possible_mila_actions:
        self.assertIn(
            original_action,
            mila_actions.mila_action_to_possible_actions(mila_action),
            f'{mila_actions} does not map to set including dm action '
            f'{human_readable_actions.action_string(original_action, None)}')

  def test_inversion_mila_actions(self):
    """Tests converting a MILA to DM to MILA action recovers original action."""
    for original_action in action_list.MILA_ACTIONS_LIST:
      possible_dm_actions = mila_actions.mila_action_to_possible_actions(
          original_action)
      for dm_action in possible_dm_actions:
        self.assertIn(
            original_action,
            mila_actions.action_to_mila_actions(dm_action),
            f'{human_readable_actions.action_string(dm_action, None)} '
            f'does not map to set including mila action {original_action}')

  def test_all_mila_actions_have_dm_action(self):
    for mila_action in action_list.MILA_ACTIONS_LIST:
      dm_actions = mila_actions.mila_action_to_possible_actions(mila_action)
      self.assertNotEmpty(dm_actions,
                          f'mila_action {mila_action} has no dm_action')

  def test_only_disband_remove_ambiguous_mila_actions(self):
    for mila_action in action_list.MILA_ACTIONS_LIST:
      dm_actions = mila_actions.mila_action_to_possible_actions(mila_action)
      if len(dm_actions) > 1:
        self.assertLen(dm_actions, 2, f'{mila_action} gives >2 dm_actions')
        orders = {action_utils.action_breakdown(dm_action)[0]
                  for dm_action in dm_actions}
        self.assertEqual(
            orders, {action_utils.REMOVE, action_utils.DISBAND},
            f'{mila_action} ambiguous but not a disband/remove action')

  def test_all_dm_actions_have_possible_mila_action_count(self):
    """DM actions correspond to possibly multiple MILA actions.

    This is because they do not specify unit type or coast when it is possible
    to infer from the board.

    There are 1, 2 or 3 possible unit descriptions (for an army and/or a fleet
    or two possible fleets in a bicoastal province) and up to 2 units specified
    in an action. Furthermore, no action can involve two fleets in bicoastal
    provinces, so the possible mila_action counts are 1, 2, 3, 4, or 6.
    """
    for action in action_list.POSSIBLE_ACTIONS:
      mila_actions_list = mila_actions.action_to_mila_actions(action)
      self.assertIn(
          len(mila_actions_list), {1, 2, 3, 4, 6},
          f'action {action} gives {len(mila_actions_list)} '
          'mila_actions, which cannot be correct')

  def test_expected_number_missing_mila_actions(self):
    """Tests MILA actions misses no actions except known convoy-related ones.

    The Mila actions list does not allow long convoys, or include any convoy
    actions that cannot affect the adjudication (e.g. ADR C ALB-TUN)

    We test these explain every situation where the actions we make are not in
    action_list.MILA_ACTIONS_LIST.
    """
    mila_actions_to_dm_actions = collections.defaultdict(list)
    long_convoys = set()

    for action in action_list.POSSIBLE_ACTIONS:
      mila_action_list = mila_actions.action_to_mila_actions(action)
      for mila_action in mila_action_list:
        mila_actions_to_dm_actions[mila_action].append(action)

        if mila_action not in action_list.MILA_ACTIONS_LIST:
          order, p1, p2, p3 = action_utils.action_breakdown(action)
          if order == action_utils.CONVOY_TO:
            long_convoys.add((p1, p2))

    reasons_for_illegal_mila_action = {
        'Long convoy to': 0,
        'Long convoy': 0,
        'Other convoy': 0,
        'Support long convoy to': 0,
        'Support alternative convoy too long': 0,
        'Unknown': 0,
    }
    for mila_action in mila_actions_to_dm_actions:
      if mila_action not in action_list.MILA_ACTIONS_LIST:
        deepmind_action = mila_actions_to_dm_actions[mila_action][0]
        order, p1, p2, p3 = action_utils.action_breakdown(deepmind_action)

        if order == action_utils.CONVOY_TO:
          # Manually checked that all of these are just long convoys (and
          # otherwise are well formatted actions)
          reasons_for_illegal_mila_action['Long convoy to'] += 1
        elif order == action_utils.CONVOY:
          if (p3, p2) in long_convoys:
            reasons_for_illegal_mila_action['Long convoy'] += 1
            continue
          else:
            # Manually checked, these are all well formatted.
            # They are irrelevant convoys, e.g. ADR C ALB-TUN
            # or they are relevant but only on long routes,
            # e.g. F IRI C StP-LVP, which is only relevant on the long route
            # BAR-NWG-NTH-ECH-IRI
            reasons_for_illegal_mila_action['Other convoy'] += 1
            continue
        elif order == action_utils.SUPPORT_MOVE_TO:
          if (p3, p2) in long_convoys:
            reasons_for_illegal_mila_action['Support long convoy to'] += 1
          else:
            # These have all been checked manually. What's happening is
            # something like F NAO S A StP - LVP. Mila's convoy rules mean that
            # the only way they allow this convoy is if NAO is part of the
            # route. The original game allows a longer convoy route, and so the
            # support is valid
            reasons_for_illegal_mila_action[
                'Support alternative convoy too long'] += 1
        else:
          reasons_for_illegal_mila_action['Unknown'] += 1

    expected_counts = {'Long convoy to': 374,
                       'Long convoy': 4238,
                       'Other convoy': 2176,
                       'Support long convoy to': 2565,
                       'Support alternative convoy too long': 27,
                       'Unknown': 0}

    self.assertEqual(reasons_for_illegal_mila_action, expected_counts,
                     'unexpected number of actions not in MILA list')


if __name__ == '__main__':
  absltest.main()

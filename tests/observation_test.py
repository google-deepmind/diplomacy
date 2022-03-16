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

"""Tests for observation format."""

import abc
import collections
import functools
from typing import Any, Dict, Sequence, Tuple

from absl.testing import absltest
import numpy as np
import tree

from diplomacy.environment import diplomacy_state
from diplomacy.environment import game_runner
from diplomacy.environment import observation_utils as utils
from diplomacy.network import config
from diplomacy.network import network_policy
from diplomacy.network import parameter_provider


def construct_observations(obs: collections.OrderedDict) -> utils.Observation:
  """Reconstructs utils.Observations from base-types.

  Reference observations are provided in observations.npz using base-types
  and numpy arrays only. This is so that users can load and inspect the file's
  content. This method reconstructs the Observation tuple required by our tests.

  Args:
    obs: element of the sequence contained in observations.npz.
  Returns:
    reconstructed Observation tuple expected in our tests.
  """
  obs['season'] = utils.Season(obs['season'])
  return utils.Observation(**obs)


def sort_last_moves(
    obs: Sequence[utils.Observation]) -> Sequence[utils.Observation]:
  """Sort the last moves observation to make test permutation invariant."""
  return [utils.Observation(o.season, o.board, o.build_numbers,
                            sorted(o.last_actions)) for o in obs]


class FixedPlayPolicy(network_policy.Policy):

  def __init__(
      self,
      actions_outputs: Sequence[Tuple[Sequence[Sequence[int]], Any]]
  ) -> None:
    self._actions_outputs = actions_outputs
    self._num_actions_calls = 0

  def __str__(self) -> str:
    return 'FixedPlayPolicy'

  def reset(self) -> None:
    pass

  def actions(
      self, slots_list: Sequence[int], observation: utils.Observation,
      legal_actions: Sequence[np.ndarray]
  ) -> Tuple[Sequence[Sequence[int]], Any]:
    del slots_list, legal_actions  # unused.
    action_output = self._actions_outputs[self._num_actions_calls]
    self._num_actions_calls += 1
    return action_output


class ObservationTest(absltest.TestCase, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def get_diplomacy_state(self) -> diplomacy_state.DiplomacyState:
    pass

  @abc.abstractmethod
  def get_parameter_provider(self) -> parameter_provider.ParameterProvider:
    """Loads params.npz and returns ParameterProvider based on its content.

    A sample implementation is as follows:

    ```
    def get_parameter_provider(self) -> parameter_provider.ParameterProvider:
      with open('path/to/sl_params.npz', 'rb') as f:
        provider = parameter_provider.ParameterProvider(f)
      return provider
    ```
    """
    pass

  @abc.abstractmethod
  def get_reference_observations(self) -> Sequence[collections.OrderedDict]:
    """Loads and returns the content of observations.npz.

    A sample implementation is as follows:

    ```
    def get_reference_observations(self) -> Sequence[collections.OrderedDict]:
      with open('path/to/observations.npz', 'rb') as f:
        observations = dill.load(f)
      return observations
    ```
    """
    pass

  @abc.abstractmethod
  def get_reference_legal_actions(self) -> Sequence[np.ndarray]:
    """Loads and returns the content of legal_actions.npz.

    A sample implementation is as follows:

    ```
    def get_reference_observations(self) -> Sequence[collections.OrderedDict]:
      with open('path/to/legal_actions.npz', 'rb') as f:
        legal_actions = dill.load(f)
      return legal_actions
    ```
    """
    pass

  @abc.abstractmethod
  def get_reference_step_outputs(self) -> Sequence[Dict[str, Any]]:
    """Loads and returns the content of step_outputs.npz.

    A sample implementation is as follows:

    ```
    def get_reference_step_outputs(self) -> Sequence[Dict[str, Any]]:
      with open('path/to/step_outputs.npz', 'rb') as f:
        step_outputs = dill.load(f)
      return step_outputs
    ```
    """
    pass

  @abc.abstractmethod
  def get_actions_outputs(
      self
  ) -> Sequence[Tuple[Sequence[Sequence[int]], Any]]:
    """Loads and returns the content of actions_outputs.npz.

    A sample implementation is as follows:

    ```
    def get_reference_observations(self) -> Sequence[collections.OrderedDict]:
      with open('path/to/actions_outputs.npz', 'rb') as f:
        actions_outputs = dill.load(f)
      return actions_outputs
    ```
    """
    pass

  def test_network_play(self):
    """Tests network loads correctly by playing 10 turns of a Diplomacy game.

    A failure of this test might mean that any of the following are true:
    1. The behavior of the user's DiplomacyState does not match our internal
    Diplomacy adjudicator. If so, test_fixed_play will also fail.
    2. The network loading is incorrect.
    """

    network_info = config.get_config()
    provider = self.get_parameter_provider()
    network_handler = parameter_provider.SequenceNetworkHandler(
        network_cls=network_info.network_class,
        network_config=network_info.network_kwargs,
        parameter_provider=provider,
        rng_seed=42)

    network_policy_instance = network_policy.Policy(
        network_handler=network_handler,
        num_players=7,
        temperature=0.2,
        calculate_all_policies=True)
    fixed_policy_instance = FixedPlayPolicy(self.get_actions_outputs())

    trajectory = game_runner.run_game(state=self.get_diplomacy_state(),
                                      policies=(fixed_policy_instance,
                                                network_policy_instance),
                                      slots_to_policies=[0] * 7,
                                      max_length=10)
    tree.map_structure(
        np.testing.assert_array_equal,
        sort_last_moves([construct_observations(o)
                         for o in self.get_reference_observations()]),
        sort_last_moves(trajectory.observations))

    tree.map_structure(np.testing.assert_array_equal,
                       self.get_reference_legal_actions(),
                       trajectory.legal_actions)

    tree.map_structure(
        functools.partial(np.testing.assert_array_almost_equal, decimal=5),
        self.get_reference_step_outputs(),
        trajectory.step_outputs)

  def test_fixed_play(self):
    """Tests the user's implementation of a Diplomacy adjudicator.

    A failure of this test indicates that the behavior of the user's
    DiplomacyState does not match our internal Diplomacy adjudicator.
    """

    policy_instance = FixedPlayPolicy(self.get_actions_outputs())
    trajectory = game_runner.run_game(state=self.get_diplomacy_state(),
                                      policies=(policy_instance,),
                                      slots_to_policies=[0] * 7,
                                      max_length=10)
    tree.map_structure(
        np.testing.assert_array_equal,
        sort_last_moves([construct_observations(o)
                         for o in self.get_reference_observations()]),
        sort_last_moves(trajectory.observations))

    tree.map_structure(np.testing.assert_array_equal,
                       self.get_reference_legal_actions(),
                       trajectory.legal_actions)

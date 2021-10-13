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

"""Parameter provider for JAX networks."""
import io
from typing import Any, Dict, Optional, Tuple

from absl import logging
import dill
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tree

from diplomacy.environment import action_utils
from diplomacy.environment import tree_utils


def apply_unbatched(f, *args, **kwargs):
  batched = f(*tree_utils.tree_expand_dims(args),
              **tree_utils.tree_expand_dims(kwargs))
  return tree.map_structure(lambda arr: np.squeeze(arr, axis=0), batched)


def fix_waives(action_list):
  """Fix action_list so that there is at most one waive, with that at the end.

  This means that build lists are invariant to order, and can be a fixed length.

  Args:
    action_list: a list of actions.

  Returns:
    A copy of action_list, modified so that any waives are truncated to 1
    and moved to the end.
  """
  non_waive_actions = [a for a in action_list if not action_utils.is_waive(a)]
  waive_actions = [a for a in action_list if action_utils.is_waive(a)]
  if waive_actions:
    # Return non-waives, then one waive.
    return non_waive_actions + waive_actions[:1]
  return non_waive_actions


def fix_actions(actions_lists):
  """Fixes network action outputs to be compatible with game_runners.

  Args:
    actions_lists: Actions for all powers in a single board state (i.e. output
    of a single inference call). Note that these are shrunk actions
    (see action_utils.py).

  Returns:
    A sanitised actions_lists suitable for stepping the environment
  """
  non_zero_actions = []
  for single_power_actions in actions_lists:
    non_zero_actions.append([])
    for unit_action in single_power_actions:
      if unit_action != 0:
        non_zero_actions[-1].append(
            action_utils.POSSIBLE_ACTIONS[unit_action >> 16])

  # Fix waives.
  final_actions = [
      fix_waives(single_power_actions)
      for single_power_actions in non_zero_actions
  ]
  return final_actions


class ParameterProvider:
  """Loads and exposes network params that have been saved to disk."""

  def __init__(self, file_handle: io.IOBase):
    self._params, self._net_state, self._step = dill.load(file_handle)

  def params_for_actor(self) -> Tuple[hk.Params, hk.Params, jnp.ndarray]:
    """Provides parameters for a SequenceNetworkHandler."""
    return self._params, self._net_state, self._step


class SequenceNetworkHandler:
  """Plays Diplomacy with a Network as policy.

  This class turns a Network into a Diplomacy bot by forwarding observations,
  and receiving policy outputs. It handles the network parameters, batching and
  observation processing.
  """

  def __init__(self,
               network_cls,
               network_config: Dict[str, Any],
               rng_seed: Optional[int],
               parameter_provider: ParameterProvider):
    if rng_seed is None:
      rng_seed = np.random.randint(2**16)
      logging.info("RNG seed %s", rng_seed)
    self._rng_key = jax.random.PRNGKey(rng_seed)
    self._network_cls = network_cls
    self._network_config = network_config

    self._rng_key, subkey = jax.random.split(self._rng_key)
    self._observation_transformer = network_cls.get_observation_transformer(
        network_config, subkey
    )

    def transform(fn_name, static_argnums=()):

      def fwd(*args, **kwargs):
        net = network_cls(**network_config)
        fn = getattr(net, fn_name)
        return fn(*args, **kwargs)

      apply = hk.transform_with_state(fwd).apply
      return jax.jit(apply, static_argnums=static_argnums)

    self._parameter_provider = parameter_provider
    # The inference method of our Network does not modify its arguments, Jax can
    # exploit this information when jitting this method to make it more
    # efficient. Network.inference takes 4 arguments.
    self._network_inference = transform("inference", 4)
    self._network_shared_rep = transform("shared_rep")
    self._network_initial_inference = transform("initial_inference")
    self._network_step_inference = transform("step_inference")
    self._network_loss_info = transform("loss_info")
    self._params = None
    self._state = None
    self._step_counter = -1

  def reset(self):
    if self._parameter_provider:
      learner_state = self._parameter_provider.params_for_actor()
      (self._params, self._state, self._step_counter) = learner_state

  def _apply_transform(self, transform, *args, **kwargs):
    self._rng_key, subkey = jax.random.split(self._rng_key)
    output, unused_state = transform(
        self._params, self._state, subkey, *args, **kwargs)
    return tree.map_structure(np.asarray, output)

  def batch_inference(self, observation, num_copies_each_observation=None):
    """Do inference on unbatched observation and state."""
    if num_copies_each_observation is not None:
      # Cast this to a tuple so that static_argnums recognises it as unchanged
      # and doesn't recompile.
      num_copies_each_observation = tuple(num_copies_each_observation)
    initial_output, step_output = self._apply_transform(
        self._network_inference, observation, num_copies_each_observation)
    final_actions = [
        fix_actions(single_board_actions)
        for single_board_actions in step_output["actions"]
    ]

    return (initial_output, step_output), final_actions

  def compute_losses(self, *args, **kwargs):
    return self._apply_transform(self._network_loss_info, *args, **kwargs)

  def inference(self, *args, **kwargs):
    outputs, (final_actions,) = apply_unbatched(self.batch_inference, *args,
                                                **kwargs)
    return outputs, final_actions

  def batch_loss_info(self, step_types, rewards, discounts, observations,
                      step_outputs):
    return tree.map_structure(
        np.asarray,
        self._network_loss_info(step_types, rewards, discounts, observations,
                                step_outputs))

  @property
  def step_counter(self):
    return self._step_counter

  def observation_transform(self, *args, **kwargs):
    return self._observation_transformer.observation_transform(
        *args, **kwargs)

  def zero_observation(self, *args, **kwargs):
    return self._observation_transformer.zero_observation(*args, **kwargs)

  def observation_spec(self, num_players):
    return self._observation_transformer.observation_spec(num_players)

  def variables(self):
    return self._params

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

"""Utility functions for tree-related operations."""

import numpy as np
import tree


def _tree_apply_over_list(list_of_trees, fn):
  """Equivalent to fn, but works on list-of-trees.

  Transforms a list-of-trees to a tree-of-lists, then applies `fn`
  to each of the inner lists.

  It is assumed that all trees have the same structure. Elements of the tree may
  be None, in which case they are ignored, i.e. they do not form part of the
  stack. This is useful when stacking agent states where parts of the state tree
  have been filtered.

  Args:
    list_of_trees: A Python list of trees.
    fn: the function applied on the list of leaves.

  Returns:
    A tree-of-arrays, where the arrays are formed by `fn`ing a list.
  """
  # The implementation below needs at least one element to infer the tree
  # structure. Check for empty inputs.
  if ((isinstance(list_of_trees, np.ndarray) and list_of_trees.size == 0) or
      (not isinstance(list_of_trees, np.ndarray) and not list_of_trees)):
    raise ValueError(
        "Expected `list_of_trees` to have at least one element but it is empty "
        "(or None).")
  list_of_flat_trees = [tree.flatten(n) for n in list_of_trees]
  flat_tree_of_stacks = []
  for position in range(len(list_of_flat_trees[0])):
    new_list = [flat_tree[position] for flat_tree in list_of_flat_trees]
    new_list = [x for x in new_list if x is not None]
    flat_tree_of_stacks.append(fn(new_list))
  return tree.unflatten_as(
      structure=list_of_trees[0], flat_sequence=flat_tree_of_stacks)


def tree_stack(list_of_trees, axis=0):
  """Equivalent to np.stack, but works on list-of-trees.

  Transforms a list-of-trees to a tree-of-lists, then applies `np.stack`
  to each of the inner lists.

  It is assumed that all trees have the same structure. Elements of the tree may
  be None, in which case they are ignored, i.e. they do not form part of the
  stack. This is useful when stacking agent states where parts of the state tree
  have been filtered.

  Args:
    list_of_trees: A Python list of trees.
    axis: Optional, the `axis` argument for `np.stack`.

  Returns:
    A tree-of-arrays, where the arrays are formed by `np.stack`ing a list.
  """
  return _tree_apply_over_list(list_of_trees, lambda l: np.stack(l, axis=axis))


def tree_expand_dims(tree_of_arrays, axis=0):
  """Expand dimension along axis across `tree_of_arrays`."""
  return tree.map_structure(lambda arr: np.expand_dims(arr, axis),
                            tree_of_arrays)

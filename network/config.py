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

"""Constants for reference_checkpoint."""

from ml_collections import config_dict

from diplomacy.environment import province_order
from diplomacy.network import network


def get_config() -> config_dict.ConfigDict:
  """Returns network config."""
  config = config_dict.ConfigDict()
  config.network_class = network.Network
  config.network_kwargs = config_dict.create(
      rnn_ctor=network.RelationalOrderDecoder,
      rnn_kwargs=dict(
          adjacency=network.normalize_adjacency(
              province_order.build_adjacency(
                  province_order.get_mdf_content(
                      province_order.MapMDF.STANDARD_MAP))),
          filter_size=64,
          num_cores=4,
      ),
      name="delta",
      num_players=7,
      area_mdf=province_order.MapMDF.BICOASTAL_MAP,
      province_mdf=province_order.MapMDF.STANDARD_MAP,
      is_training=False,
      shared_filter_size=160,
      player_filter_size=160,
      num_shared_cores=12,
      num_player_cores=3,
      value_mlp_hidden_layer_sizes=(256,),
      actions_since_last_moves_embedding_size=10)
  return config

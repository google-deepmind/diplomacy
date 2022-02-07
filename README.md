# README

This directory contains code to let agents from [Learning to Play No Press
Diplomacy with Best Response Policy Iteration (Anthony et al
2020)](https://arxiv.org/abs/2006.04635) play Diplomacy.

The code provided here, paired with a Diplomacy environment and adjudicator, can
be used to evaluate our agents, and generate game trajectories.

A Diplomacy environment/adjudicator is required to play games, specifications
for this module can be found in the protocol in
`environment/diplomacy_state.py`.

This readme describes the observations and action space required, and tests to
confirm the environment and agent are working correctly.

## Implementation Details

### Action Space

In Diplomacy, each turn a player must choose actions for each of their units.

The unit-actions always have an order type (like move or support); always have a
source area (where the unit is now); usually have a target area (e.g. the
destination of a movement). Support move and convoy order types have a third
area, which is the location of the unit receiving support/being convoyed.

The unit-actions are represented by a 64 bit integer. Bits 0-31 represent
ORDER|ORDERED AREA|TARGET AREA|THIRD AREA, (each of these takes up to 8 bits).
Bits 32-47 are always 0. Bits 48-63 are used to record the index of each action
into POSSIBLE_ACTIONS.

The different order codes are constants can be found in
`environment/action_utils.py`.

The 8-bit representation of the areas in the action are as follows:

*    The first 7 bits identify the province. The ids of each province are given
     by calling `province_order.province_name_to_id()`

*    The last bit is a coast flag to identify which coast of a bi-coastal
     province is being referred to. It is 1 for the South Coast area. For the
     main area, single-coastal provinces, or the North/East coast of a
     bi-coastal province, it is 0

(Note: elsewhere in the code areas are represented as a (province_id, coast_id)
tuple, where coast_id is 0 for the main area and 1 or 2 for the two coasts, or
as a single area_id from 0 to 80.)

Bits 0-31 make the meaning of an action easy to calculate. The file
`environment/actions_utils.py` includes several functions for parsing unit
actions. The file `environment/human_readable_actions.py` converts the integer
actions into a human readable format.

The indexing part of the action representation is used to convert between the
one-hot output of a neural network and the interpretable action representation.

Not all syntactically-correct unit-actions are possible in Diplomacy, for
instance Army Paris Move to Berlin is never legal because Berlin is not adjacent
to Paris. The list of actions in `environment/action_list.py` contains all
actions that could ever be legal in a game of Diplomacy. This list allows the
full 64 bit action to be recovered from the action’s index.

The file `environment/mila_actions.py` contains functions to convert between the
action format used by this codebase (hereafter DM actions) and the action format
used by Pacquette et al. (MILA actions)

These mappings are not one-to-one for a few reasons: - MILA actions do not
distinguish between disbanding a unit in a retreats phase and disbanding during
the builds phase, DM actions do. - MILA actions specify the unit type
(fleet/army) and coast it occupies when referring to units on the board. DM
actions specify these details only for build actions. In all other circumstances
the province uniquely specifies the unit given the context of the board state. -
Pacquette et al. disallowed long convoys, and some convoy orders that are always
irrelevant to the adjudicaiton.

For converting from MILA actions to DM actions, the function
`mila_action_to_action` gives a one-to-one conversion by taking the current
season (an `environment/observation_utils.Season`) as additional context.

When converting from DM actions to MILA actions, the function
`action_to_mila_actions` returns a list of up to 6 possible MILA actions. Given
a state, at most one of these actions can be legal, which one can be inferred by
checking the game state.

### Observations

The observation format is defined in `observation_utils.Observation`. It is a
named tuple of:

season: One of `observation_utils.Season`

board: An array of shape (`observation_utils.NUM_AREAS`,
`utils.PROVINCE_VECTOR_LENGTH`). The areas are ordered by their AreaID as given
by `province_order.province_name_to_id(province_order.MapMDF.BICOASTAL_MAP)`.
The vector representing a single area is, in order:
- 3 flags representing the presence of an army, a fleet or an empty province
respectively
- 7 flags representing the owner of the unit, plus an 8th that is true if there
is no such unit
- 1 flag representing whether a unit can be built in the province
- 1 flag representing whether a unit can be removed from the province
- 3 flags representing the existence of a dislodged army or fleet, or no
dislodged unit
- 7 flags representing the owner of the dislodged unit, plus an 8th that is true
if there is no such unit
- 3 flags representing whether the area is a land, sea or coast area of a
bicoastal province. These are mutually exclusive: a land area is any area an
army can occupy, which includes e.g. StP but does not include StP/NC or StP/SC.
- 7 flags representing the owner of the supply centre in the province, plus an
8th representing an unowned supply centre. The 8th flag is false if there is no
SC in the area

build_numbers: A vector of length 7 saying how many units a player may build
(positive values) or must remove (negative values)

last_actions: A list of the actions submitted in the last phase of the game.
They are in the same order as given in the previous step method, but flattened
into a single list.

For the build_numbers, last_actions, and one-hot flags of unit and supply centre
owners, the powers are ordered alphabetically: Austria, England, France,
Germany, Italy, Russia, Turkey.

## Run network test

You can make sure this code runs successfully by using the `run.sh` script
provided. The script will set up a fresh virtual environment, download the
appropriate libraries, and then run our `tests/network_test.py` (see below).

You can also do these steps manually using the following commants:

### Setup

To set up a ptyhon3 virtual environment with the required dependencies, use the
following commands, or simply run `run.sh`.

```shell
cd ..
python3 -m venv dip_env
source dip_env/bin/activate
pip3 install --upgrade pip
pip3 install -r diplomacy/requirements.txt
```

### Running a basic smoke test

Use the following command to run basic tests and make sure you have all the
required dependencies. See the next paragraph for an more detailed explanation
of the tests we provide.

```shell
python3 -m diplomacy.tests.network_test
```

## Tests

We provide two test files:

*   `tests/network_test.py` contains smoke tests that will fail if the network
    does not produce the correct output shape or format, or is unable to perform
    a dummy parameter update.

*   `tests/observation_test.py` tests that the network plays Diplomacy as
    expected given the paremeters we provide, and it checks that the user's
    Diplomacy environment and adjudicator produce the same observations and
    trajectories as our internal implementation. Specifically, this file
    contains a template test class which the user must complete with simple
    methods to load the parameters and trajectory files we provide alongside
    this codebase. Suggestions for how to implement those are provided in the
    file itself.

## Download parameters and test trajectories.

We provide network parameters for the SL and FPPI-2 training schemes (see
[Learning to Play No Press Diplomacy with Best Response Policy Iteration
(Anthony et al 2020)](https://arxiv.org/abs/2006.04635)).

We further provide trajectories generated with the SL parameters and our
internal Diplomacy environment and adjudicator. This is so that users can verify
that the network plays Diplomacy as expected, and that their environment and
adjudicator produce match the behavior of our internal ones using the tests
described above.

| Type | Description | Link |
|---|---|---|
| Parameters | Supervised Imitation Learning (SL) | [download](https://storage.googleapis.com/dm-diplomacy/sl_params.npz) |
| Parameters | Fictitious Play Policy Iteration 2 (FPPI-2) | [download](https://storage.googleapis.com/dm-diplomacy/fppi2_params.npz) |
| Trajectory | Observations | [download](https://storage.googleapis.com/dm-diplomacy/observations.npz)|
| Trajectory | Legal Actions | [download](https://storage.googleapis.com/dm-diplomacy/legal_actions.npz)|
| Trajectory | Step Outputs | [download](https://storage.googleapis.com/dm-diplomacy/step_outputs.npz)|
| Trajectory | Action Outputs | [download](https://storage.googleapis.com/dm-diplomacy/actions_outputs.npz)|

## Citing

Please cite [Learning to Play No Press Diplomacy with Best Response Policy
Iteration (Anthony et al 2020)](https://arxiv.org/abs/2006.04635)

```
@misc{anthony2020learning,
  title={Learning to Play No-Press Diplomacy with Best Response Policy Iteration},
  author={Thomas Anthony and Tom Eccles and Andrea Tacchetti and János Kramár
  and Ian Gemp and Thomas C. Hudson and Nicolas Porcel and Marc Lanctot and
  Julien Pérolat and Richard Everett and Roman Werpachowski and Satinder Singh
  and Thore Graepel and Yoram Bachrach},
   year={2020},
   eprint={2006.04635},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}
```

## Disclaimer

This is not an official Google product.

# Understanding Disentangled RNNs

## General Notes
- The original implementation of DisRNNs and agents can be found here: https://github.com/google-deepmind/disentangled_rnns. Unfortunately, I found the repository only after implementing my version.
- The original implementation of the belief analysis (the code from the Henning Paper) can be found here: https://github.com/mobeets/value-rnn-beliefs.
- My current version of DisRNNs uses a for-loop to iterate over the hidden state variables and update them. I suspect the training process could be accelerated if the for-loop is replaced with a "parallelized JAX call" (although I'm not sure if it is already being optimized).

## Structure of Project
### Sandboxes
Three different sandboxes are utilized in this project:
- `Sandbox_CogModelling.ipynb`: Contains code related to the training and analysis of DisRNNs in the modeling task.
- `Sandbox_ValueModelling.ipynb`: Contains all code related to training models on the Starkweather experiment data and the subsequent analysis.
- `Sandbox_General.ipynb`: Used for creating the Agents-Datasets as well as for code not related to the above topics.

### `rl_env` directory
The `rl_env` directory contains all code related to the Leaky Actor-Critic Agent, the Q-Learning Agent, and the environment in which they operate. The analysis of their update rules is also located in this directory. For both agents, an `analyze...` method (for plotting the update rules) and a `plot...` method (for plotting an example session) are located in the `agent_analysis.py` file.

### `models` directory
This directory contains all code related to the GRU-RNN and the disentangled RNN, as well as the necessary training code and their analysis. The definitions for both model types are located in the respective `...model` files, while helper functions specific to one of the two models are found in the corresponding `...utils` files. The `rnn_utils.py` file contains code that is not specific to either RNN but is used, for example, in training both models.

### `belief_...` files
These files contain code related to the POMDP and the analysis of the belief state.


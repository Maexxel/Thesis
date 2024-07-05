from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Dict, List
if TYPE_CHECKING:
    from .disrnn_utils import DisRNNTrainState
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

from .rnn_utils import load_model_state


import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

def plot_bottlenecks(params: Dict,
                     sort_latents: bool=True,
                     obs_names: None | List[str] =None):
    """
    Source:
        Adapted from https://github.com/kstach01/CogModelingRNNsTutorial.git

    Plot the bottleneck sigmas from an DisRNN.

    Args:
        params (dict): Dictionary containing model parameters.
        sort_latents (bool, optional): Whether to sort latent sigmas. Default is True.
        obs_names (list or None, optional): List of observation names. Default is None.

    Returns:
        matplotlib.figure.Figure: Figure object containing the plotted subplots.
    """
    params_disrnn = params['DisRNNCell0']
    latent_dim = params_disrnn['latent_bottleneck_sigmas'].shape[0]
    
    if obs_names is None:
        obs_names = ['Choice', 'Reward']
    obs_dim = len(obs_names)

    latent_sigmas = 2 * jax.nn.sigmoid(
        jnp.array(params_disrnn['latent_bottleneck_sigmas'])
    )

    update_sigmas = 2 * jax.nn.sigmoid(
        jnp.array(params_disrnn['update_bottleneck_sigmas'])
    )
    update_sigmas = update_sigmas.reshape((latent_dim, -1))

    if sort_latents:
        latent_sigma_order = np.argsort(
            params_disrnn['latent_bottleneck_sigmas']
        )
        latent_sigmas = latent_sigmas[latent_sigma_order]
        update_sigma_order = np.concatenate(
            (np.arange(0, obs_dim) + latent_dim, latent_sigma_order), axis=0
        )
        update_sigmas = update_sigmas[latent_sigma_order, :]
        update_sigmas = update_sigmas[:, update_sigma_order]

    latent_names = np.arange(1, latent_dim + 1)
    fig, _ = plt.subplots(1, 2, figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(np.swapaxes([1 - latent_sigmas], 0, 1), cmap='Oranges')
    plt.clim(vmin=0, vmax=1)
    plt.yticks(ticks=range(latent_dim), labels=latent_names)
    plt.xticks(ticks=[])
    plt.ylabel('Latent #')
    plt.title('Latent Bottlenecks')

    plt.subplot(1, 2, 2)
    plt.imshow(1 - update_sigmas, cmap='Oranges')
    plt.clim(vmin=0, vmax=1)
    plt.colorbar()
    plt.yticks(ticks=range(latent_dim), labels=latent_names)
    xlabels = np.concatenate((np.array(obs_names), latent_names))
    plt.xticks(
        ticks=range(len(xlabels)),
        labels=xlabels,
        rotation='vertical',
    )
    plt.ylabel('Latent #')
    plt.title('Update MLP Bottlenecks')
    
    return fig


import absl.logging
absl.logging.set_verbosity('error')

def plot_bottleneck_evolution(state: DisRNNTrainState, path: str, end=None):
    """
    Plots the evolution of latent and update sigmas over training checkpoints.

    Args:
        state (DisRNNTrainState): The state of the DisRNN training.
        path (str): Path to the directory containing checkpoint files.
        end (int, optional): Number of checkpoints to consider. Defaults to None, 
            which includes all checkpoints found in the path.

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib Figure object containing the plot.
    """
    checkpoints_raw = os.listdir(path)

    def map_func(filename: str) -> Tuple[str, int]:
        """
        Extracts and parses epoch number from checkpoint filenames.

        Args:
            filename (str): Filename of the checkpoint.

        Returns:
            Tuple[str, int]: Tuple containing prefix and epoch number.
        """
        try:
            str_, epoch = filename.split('_')
            epoch = int(epoch)
            return str_, epoch
        except Exception as e:
            print(f"<{filename}> is not a valid checkpoint-dir filename.")
            return None

    def filter_func(dir_tuple: Tuple[str, int] | None) -> bool:
        """
        Filters out invalid checkpoint tuples.

        Args:
            dir_tuple (Tuple[str, int] | None): Tuple containing prefix and epoch number.

        Returns:
            bool: True if the tuple is valid and prefix is 'checkpoint', False otherwise.
        """
        return dir_tuple is not None and dir_tuple[0] == "checkpoint"

    checkpoints = sorted(filter(filter_func, map(map_func, checkpoints_raw)), key=lambda x: x[1])

    end = len(checkpoints) if end is None else end
    checkpoints = checkpoints[:end]

    latent_sigmas = []
    update_sigmas = []

    for checkpoint in checkpoints:
        state = load_model_state(state, path, step=checkpoint[1])
        if state is None:
            print(f"Checkpoint <checkpoint_{checkpoint[1]}> could not be loaded.")
            continue

        disrnn_params = state["params"]["DisRNNCell0"]
        latent_sigmas.append(disrnn_params["latent_bottleneck_sigmas"])
        update_sigmas.append(disrnn_params["update_bottleneck_sigmas"])

    latent_sigmas = np.stack(latent_sigmas)
    latent_sigmas = np.array(2 * jax.nn.sigmoid(latent_sigmas))
    latent_sort_idx = np.argsort(latent_sigmas[-1, :])
    latent_sigmas = latent_sigmas[:, latent_sort_idx].T

    update_sigmas = np.stack(update_sigmas)
    update_sigmas = np.array(2 * jax.nn.sigmoid(update_sigmas))
    epochs, latent_dim, in_dim = len(latent_sigmas[0]), len(latent_sigmas), 2

    update_sigmas = update_sigmas.reshape(epochs, latent_dim, latent_dim + in_dim)
    update_sigmas_sort = np.hstack([np.arange(in_dim) + latent_dim, latent_sort_idx])
    update_sigmas = update_sigmas[:, latent_sort_idx, :]
    update_sigmas = update_sigmas[:, :, update_sigmas_sort]
    update_sigmas = update_sigmas.reshape(epochs, latent_dim * (latent_dim + in_dim))
    update_sigmas = update_sigmas.T

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True, height_ratios=[1, 5])

    # Plot latent_sigmas (first subplot)
    im1 = ax1.imshow(1 - latent_sigmas, cmap='Oranges', aspect="auto", vmin=0, vmax=1,
                     extent=(-.5, checkpoints[-1][1] + .5, -.5, latent_dim + .5))
    ax1.set_title('Latent Sigmas')
    ax1.set_xlabel('Epoch')
    ax1.set_yticks(np.arange(0, 5, 1))
    ax1.set_yticklabels([f"Latent {i}" for i in range(1, latent_dim + 1)])

    # Plot update_sigmas (second subplot)
    im2 = ax2.imshow(1 - update_sigmas, cmap='Oranges', aspect="auto", vmin=0, vmax=1,
                     extent=(-.5, checkpoints[-1][1] + .5, len(update_sigmas) + .5, -.5))
    ax2.set_title('Update Sigmas')
    ax2.set_xlabel('Epoch')
    ax2.set_yticks(np.arange(-.5, 34.5, 7))
    ax2.set_yticklabels([f"Latent {i}" for i in range(1, latent_dim + 1)])
    
    # Adjust y-labels of second plot
    plt.setp(ax2.yaxis.get_majorticklabels(), rotation=90)
    dx = -2 / 3
    dy = -20 / 72
    offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)

    for label in ax2.yaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # Adjust tick labels
    per_latent_labels = ["Choice", "Reward"] + [f"Latent {i}" for i in range(1, latent_dim + 1)]
    ax2.set_yticks(np.arange(0, 35), minor=True)
    ax2.set_yticklabels(per_latent_labels * latent_dim, minor=True)

    # Add gridlines and adjust labels
    ax2.grid(which='major', color='k', linestyle='-', linewidth=2, axis="y")

    # Create a colorbar for both plots
    fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical')

    return fig

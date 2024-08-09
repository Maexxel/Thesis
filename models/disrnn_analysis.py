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
from .disrnn_model import DisRNNCell

def plot_bottlenecks(params: Dict,
                     sort_latents: bool=True,
                     obs_names: None | List[str]=None,
                     title: str = ""):
    """
    Source:
        Adapted from https://github.com/kstach01/CogModelingRNNsTutorial.git

    Plot the bottleneck sigmas from an DisRNN.

    Args:
        params (dict): Dictionary containing model parameters.
        sort_latents (bool, optional): Whether to sort latent sigmas. Default is True.
        obs_names (list or None, optional): List of observation names. Default is None.
        title (str, optional): Title of the plot. Default is no title.

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
    
    if title != "":
        fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    
    return fig

def plot_bottleneck_evolution(path: str, end=None):
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
    IN_DIM = 2
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
        state = load_model_state(path=path, step=checkpoint[1])
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
    epochs, latent_dim, in_dim = len(latent_sigmas[0]), len(latent_sigmas), IN_DIM
    n_update_bootlenecks = (in_dim + latent_dim) * latent_dim

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
    ax1.set_yticks(np.arange(0, latent_dim, 1))
    ax1.set_yticklabels([f"Latent {i}" for i in range(1, latent_dim + 1)])

    # Plot update_sigmas (second subplot)
    im2 = ax2.imshow(1 - update_sigmas, cmap='Oranges', aspect="auto", vmin=0, vmax=1,
                     extent=(-.5, checkpoints[-1][1] + .5, len(update_sigmas) + .5, -.5))
    ax2.set_title('Update Sigmas')
    ax2.set_xlabel('Epoch')
    ax2.set_yticks(np.arange(-.5, n_update_bootlenecks - .5, 7))
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
    ax2.set_yticks(np.arange(0, n_update_bootlenecks), minor=True)
    ax2.set_yticklabels(per_latent_labels * latent_dim, minor=True)

    # Add gridlines and adjust labels
    ax2.grid(which='major', color='k', linestyle='-', linewidth=2, axis="y")

    # Create a colorbar for both plots
    fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical')

    return fig


def decode_disrnn_params(disrnn_params: Dict) -> Tuple[int, int, int, List[int], List[int]]:
    """Decodes DisRNN parameters from the given dictionary.

    Args:
        disrnn_params (Dict): A dictionary containing DisRNN parameters.

    Returns:
        Tuple[int, int, int, List[int], List[int]]: A tuple containing:
            - hidden_size (int): The size of the hidden layer.
            - input_size (int): The size of the input layer.
            - out_size (int): The size of the output layer.
            - update_mlp_shape (List[int]): The shape of the update MLP.
            - choice_mlp_shape (List[int]): The shape of the choice MLP.

    Raises:
        AssertionError: If `hidden_size` or `input_size` is not greater than 0.
    """
    hidden_size = len(disrnn_params["latent_bottleneck_sigmas"])
    assert hidden_size > 0, "hidden_size must be greater than 0"
    
    input_size = len(disrnn_params["update_bottleneck_sigmas"]) // hidden_size - hidden_size
    assert input_size > 0, "input_size must be greater than 0"
    
    out_size = len(disrnn_params[f'Dense_{hidden_size}']["bias"])
    
    update_mlp_shape = [len(umlp["bias"]) for umlp in disrnn_params["update_mlp_0"].values()]
    choice_mlp_shape = [len(umlp["bias"]) for umlp in disrnn_params["choice_mlp"].values()]

    return hidden_size, input_size, out_size, update_mlp_shape, choice_mlp_shape


def force_carry(carries: jnp.array, xs: jnp.array, disrnn_params: Dict) -> Tuple[jnp.array, jnp.array]:
    """Forces carry operations in a DisRNN cell.

    Args:
        carries (jnp.array): The carry inputs, a 2D array of shape (batch_size, hidden_size).
        xs (jnp.array): The input data, a 2D array of shape (batch_size, input_size).
        disrnn_params (Dict): A dictionary containing DisRNN parameters.

    Returns:
        Tuple[jnp.array, jnp.array]: A tuple containing:
            - Updated carries (jnp.array): A 2D array of updated carry values of shape (batch_size, hidden_size).
            - Outputs (jnp.array): A 2D array of output values of shape (batch_size, out_size).

    Raises:
        AssertionError: If any of the input shapes or types are invalid.
    """
    # Check input types and shapes
    assert isinstance(disrnn_params, dict), "disrnn_params must be a dictionary"
    assert len(carries.shape) == len(xs.shape) == 2, "carries and xs must be 2D arrays"
    assert carries.shape[0] == xs.shape[0], "carries and xs must have the same batch size"
    
    hidden_size, input_size, out_size, update_mlp_shape, choice_mlp_shape = decode_disrnn_params(disrnn_params)

    assert carries.shape[1] == hidden_size, "carries must have shape (batch_size, hidden_size)"
    assert xs.shape[1] == input_size, "xs must have shape (batch_size, input_size)"

    # Initialize DisRNNCell
    disrnn_cell = DisRNNCell(hidden_size=hidden_size,
                             in_dim=input_size,
                             out_dim=out_size,
                             update_mlp_shape=update_mlp_shape,
                             choice_mlp_shape=choice_mlp_shape)
    
    # Define the function to be vectorized
    def func(full_input: jnp.array) -> Tuple[jnp.array, jnp.array]:
        carrie_in = full_input[:hidden_size]
        input_data = full_input[hidden_size:]
        carrie, out = disrnn_cell.apply({"params": disrnn_params},
                                        carrie_in,
                                        input_data,
                                        rngs={"bottleneck_master_key": jax.random.PRNGKey(0)})
        
        return jnp.hstack([carrie, out])
    
    # Stack carries and xs and apply the vectorized function
    full_input = np.hstack([carries, xs])
    vfunc = jax.vmap(func, in_axes=0)
    full_out = vfunc(full_input)
    
    return full_out[:, :hidden_size], full_out[:, hidden_size:]


def plot_update_rules(params: Dict) -> List[plt.Figure]:
    """Generates visualizations of the update rules of a DisRNN.

    Args:
        params (Dict): A dictionary containing the parameters of a DisRNN model.

    Returns:
        List[plt.Figure]: A list of matplotlib figures showing the update rules.

    Raises:
        AssertionError: If any parameter checks fail.
    """
    # Extract parameters from the input dictionary
    params_disrnn = params['DisRNNCell0']
    latent_dim = params_disrnn['latent_bottleneck_sigmas'].shape[0]

    # Compute sigmas for latent and update bottlenecks
    latent_sigmas = 2 * jax.nn.sigmoid(jnp.array(params_disrnn['latent_bottleneck_sigmas']))
    update_sigmas = 2 * jax.nn.sigmoid(jnp.array(params_disrnn['update_bottleneck_sigmas']))
    update_sigmas = update_sigmas.reshape((latent_dim, -1))
    latent_sigma_order = np.argsort(params_disrnn['latent_bottleneck_sigmas'])

    def plot_update_1d(params: Dict, latent_idx: int, observations: List[List[int]], titles: List[str]) -> plt.Figure:
        """Plots 1D update rules.

        Args:
            params (Dict): DisRNN parameters.
            latent_idx (int): Index of the latent variable.
            observations (List[List[int]]): List of observations.
            titles (List[str]): Titles for the plots.

        Returns:
            plt.Figure: The resulting plot figure.
        """
        # Set up general layout for 1D plot
        lim = 1
        hidden_size = len(params["latent_bottleneck_sigmas"])
        state_bins = np.linspace(-lim, lim, 20)
        colormap = plt.get_cmap('viridis', 3)
        colors = colormap.colors

        fig, ax = plt.subplots(1, len(observations), figsize=(len(observations) * 4, 5.5))
        plt.subplot(1, len(observations), 1)
        plt.ylabel('Updated Activity')

        # Prepare input data for the model
        bloated_observations = np.repeat(np.array(observations), len(state_bins), 0)
        carries = np.zeros((len(state_bins), hidden_size))
        carries[:, latent_idx] = state_bins
        bloated_carrie = np.tile(carries, [len(observations), 1])

        # Probe the model with input data
        carries, _ = force_carry(bloated_carrie, bloated_observations, params)
        carries = carries[:, latent_idx].reshape(len(observations), -1)

        # Plot the output data
        for observation_i in range(len(observations)):
            plt.subplot(1, len(observations), observation_i + 1)
            plt.plot((-3, 3), (-3, 3), '--', color='grey')
            plt.plot((-3, 3), (0, 0), color='black')
            plt.plot((0, 0), (-3, 3), color='black')
            plt.plot(state_bins, carries[observation_i], color=colors[1])
            plt.title(titles[observation_i])
            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)
            plt.xlabel('Previous Activity')

            if isinstance(ax, np.ndarray):
                ax[observation_i].set_aspect('equal')
            else:
                ax.set_aspect('equal')
        return fig

    def plot_update_2d(params: Dict, latent_idx: int, latent_input: int, observations: List[List[int]], titles: List[str]) -> plt.Figure:
        """Plots 2D update rules.

        Args:
            params (Dict): DisRNN parameters.
            latent_idx (int): Index of the latent variable.
            latent_input (int): Index of the latent input variable.
            observations (List[List[int]]): List of observations.
            titles (List[str]): Titles for the plots.

        Returns:
            plt.Figure: The resulting plot figure.
        """
        # Set up general layout for 2D plot
        lim = 1
        hidden_size = len(params["latent_bottleneck_sigmas"])
        state_bins = np.linspace(-lim, lim, 10)
        colormap = plt.get_cmap('viridis', len(state_bins))
        colors = colormap.colors

        fig, ax = plt.subplots(1, len(observations), figsize=(len(observations) * 2 + 10, 5.5))
        plt.subplot(1, len(observations), 1)
        plt.ylabel('Updated Latent ' + str(latent_idx + 1) + ' Activity')

        # Prepare input data for the model
        bloated_observations = np.repeat(np.array(observations), len(state_bins) * len(state_bins), 0)
        bloated_carries = np.zeros((len(bloated_observations), hidden_size))

        # Set up carries with state bins
        second_latent_input = np.tile(np.repeat(state_bins, len(state_bins)), len(observations))
        bloated_carries[:, latent_input] = second_latent_input
        first_latent_input = np.tile(state_bins, len(state_bins) * len(observations))
        bloated_carries[:, latent_idx] = first_latent_input

        # Probe the model with input data
        carries, _ = force_carry(bloated_carries, bloated_observations, params)
        carries = carries[:, latent_idx].reshape(len(observations), len(state_bins), len(state_bins))

        # Plot the output data
        for observation_i in range(len(observations)):
            plt.subplot(1, len(observations), observation_i + 1)
            plt.plot((-3, 3), (-3, 3), '--', color='grey')
            plt.plot((-3, 3), (0, 0), color='black')
            plt.plot((0, 0), (-3, 3), color='black')

            for si_i in np.arange(len(state_bins)):
                plt.plot(state_bins, carries[observation_i][si_i], color=colors[si_i])

            plt.title(titles[observation_i])
            plt.xlim(-lim, lim)
            plt.ylim(-lim, lim)
            plt.xlabel('Latent ' + str(latent_idx + 1) + ' Activity')

            if isinstance(ax, np.ndarray):
                ax[observation_i].set_aspect('equal')
            else:
                ax.set_aspect('equal')
        return fig

    figs = []
    for latent_i in latent_sigma_order:
        if latent_sigmas[latent_i] < 0.8:
            update_mlp_inputs = np.argwhere(update_sigmas[latent_i] < 0.9)
            choice_sensitive = np.any(update_mlp_inputs == 0 + latent_dim)
            reward_sensitive = np.any(update_mlp_inputs == 1 + latent_dim)

            # Determine observations and titles based on sensitivities
            if choice_sensitive and reward_sensitive:
                observations = ([0, 0], [0, 1], [1, 0], [1, 1])
                titles = ('Left, Unrewarded', 'Left, Rewarded', 'Right, Unrewarded', 'Right, Rewarded')
            elif choice_sensitive:
                observations = ([0, 0], [1, 0])
                titles = ('Choose Left', 'Choose Right')
            elif reward_sensitive:
                observations = ([0, 0], [0, 1])
                titles = ('Rewarded', 'Unrewarded')
            else:
                observations = ([0, 0],)
                titles = ('All Trials',)

            # Determine if the update depends on other latent values
            latent_sensitive = update_mlp_inputs[update_mlp_inputs < latent_dim]
            latent_sensitive = np.delete(latent_sensitive, latent_sensitive == latent_i)

            fig = None
            if latent_sensitive.size == 0:
                # Plot 1D update rule
                fig = plot_update_1d(params_disrnn, latent_i, observations, titles)
            else:
                # Plot 2D update rule
                fig = plot_update_2d(params_disrnn, latent_i, latent_sensitive[np.argmax(latent_sensitive)], observations, titles)
            
            if len(latent_sensitive) > 1:
                print('WARNING: This update rule depends on more than one other latent. Plotting just one of them.')

            figs.append(fig)
    
    return figs

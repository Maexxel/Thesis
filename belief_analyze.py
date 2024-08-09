#-- Source: The fit_weights and _make_predictions methods are adapted from
#-- https://github.com/mobeets/value-rnn-beliefs

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from starkweather import Trial

import jax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from custom_datasets import MyStarkweather
from belief_utils import MyTrial, RpeGroup, convert_to_mytrial
from belief_pomdp import add_pomdp_states_beliefs
from models.disrnn_utils import create_disrnn_train_state
from models.gru_utils import create_gru_train_state
from models.rnn_utils import load_model_state, eval_value_wrapper


def fit_weights(trials: List[Trial], gamma=0.93, lambda0=0):
    X = np.vstack([trial.Z for trial in trials])
    X = np.hstack([X, np.ones((X.shape[0],1))])
    r = np.vstack([trial.y for trial in trials])[1:]
    X_cur = X[:-1]
    X_next = X[1:]
    
    B_cur = X_cur
    B_next = X_cur - gamma*X_next
    
    X = B_cur.T @ B_next + lambda0*np.eye(B_cur.shape[1])
    y = B_cur.T @ r
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w[:,0]


def _make_predictions(rs, Z, w, gamma, add_bias=True):
    if add_bias:
        Z = np.hstack([Z, np.ones((Z.shape[0],1))])
    
    rpes = []
    values = []
    for t in range(1,len(rs)):
        zprev = Z[t-1,:]
        z = Z[t,:]
        rpe = rs[t] + w @ (gamma*z - zprev)
        value = w @ zprev
        
        rpes.append(rpe)
        values.append(value)
    rpes.append(np.nan)
    values.append(np.nan)
    return np.array(rpes), np.array(values)


def add_value_and_rpe(trials, value_weights, gamma):
    Z = np.vstack([trial.Z for trial in trials])
    rs = np.hstack([trial.y[:,0] for trial in trials])
    rpes, values = _make_predictions(rs, Z, value_weights, gamma)
    
    i = 0
    for trial in trials:
        trial.value = values[i:(i+trial.trial_length)]
        trial.rpe = rpes[i:(i+trial.trial_length-1)]
        i += trial.trial_length
    return trials


def calc_mse(trial_dict: Dict[str, List[MyTrial]]) -> Dict[str, float]:
    """Calculates the mean squared error (MSE) of RPEs compared to a POMDP baseline.

    Args:
        trial_dict (Dict[str, List[MyTrial]]): A dictionary where keys are trial IDs and values are 
            lists of MyTrial objects containing RPEs.

    Returns:
        Dict[str, float]: A dictionary where keys are trial IDs with "_mse" suffix and values are 
            the corresponding MSE values.
    """
    result_mses = {}
    pomdp_rpes = None
    only_rpes = {}

    for key, value in trial_dict.items():
        stacked_rpes = np.hstack([mytrial.rpes for mytrial in value])
        if "pomdp" in key:
            pomdp_rpes = stacked_rpes
        else:
            only_rpes[key] = stacked_rpes

    for key, value in only_rpes.items():
        mse = np.mean(np.square(pomdp_rpes - value))
        result_mses[key] = mse

    return result_mses


def multi_value_analyze(ommision_probability: float,
                        model_confs: Dict[str, Dict[str, int | float]],
                        models_path: str = "data/models/belief_models",
                        verbose: bool = True) -> Dict[str, List[MyTrial]]:
    """Analyzes multiple value-based RNN models and compares them to a POMDP reference model.

    Args:
        ommision_probability (float): The probability of omission events in the evaluation dataset.
        model_confs (Dict[str, Dict[str, int | float]]): Configuration dictionary for models.
            The keys are model paths and the values are dictionaries containing model configurations.
        models_path (str, optional): Path to the directory containing model files. Defaults to "data/models/belief_models".
        verbose (bool, optional): If True, prints detailed logs. Defaults to True.

    Returns:
        Dict[str, List[MyTrial]]: A dictionary where keys are model descriptions and values are lists of MyTrial objects.
    """
    DISRNN_ID = "disrnn"
    VALUE_ID = "vrnn"

    # Initialize evaluation dataset
    eval_dataset = MyStarkweather(n_sequences=2,
                                  len_sequences=10_000,
                                  omission_probability=ommision_probability,
                                  iti_p=1/8,
                                  iti_min=10)
    my_trial_dict: Dict[str, List[MyTrial]] = {}

    # Create POMDP reference model
    if verbose:
        print("Creating POMDP reference model...")
    raw_pomdp_trials = add_pomdp_states_beliefs(eval_dataset, 0.1, 1/8, 10)
    raw_pomdp_trials = add_value_and_rpe(raw_pomdp_trials, fit_weights(raw_pomdp_trials), gamma=0.93)
    my_trial_dict["pomdp"] = convert_to_mytrial(raw_pomdp_trials)

    if verbose:
        print(f"Analyzing models...")

    for model_path in model_confs.keys():
        model_conf = model_confs[model_path]

        # Check if model was trained on correct omission probability
        if ommision_probability != model_conf["ommision_probs"]:
            continue
        if verbose:
            print(f"\tFound model at path: {model_path} with conf: {', '.join(key + '=' + str(value) for key, value in model_conf.items())}")

        # Initialize model state
        master_key = jax.random.PRNGKey(0)
        if model_conf["type"] == VALUE_ID or model_conf["type"] == "vrnnuntrained":
            model_state = create_gru_train_state(
                master_key,
                learning_rate=model_conf["learning_rate"],
                hidden_size=model_conf["hidden_size"],
                batch_size=10,
                seq_length=1_000,
                out_dim=1,
                in_dim=2
            )
        elif model_conf["type"] == DISRNN_ID:
            model_state = create_disrnn_train_state(
                master_rng_key=master_key,
                learning_rate=model_conf["learning_rate"],
                hidden_size=model_conf["hidden_size"],
                batch_size=10,
                seq_length=1000,
                in_dim=2,
                out_dim=1,
                update_mlp_shape=[5, 5, 5],
                choice_mlp_shape=[2, 2],
                kl_loss_factor=model_conf["kl_loss"]
            )
        else:
            raise TypeError(f"Invalid model type <{model_conf['type']}> encountered.")
        
        model_state = load_model_state(path=f"{models_path}/{model_path}", state=model_state)
        if verbose:
            print("\t\tModel state loaded.")

        # Evaluate model
        model_trials = eval_value_wrapper(eval_dataset, model_state, sigma=0.05, true_output=1)
        if verbose:
            print("\t\tModel probed.")

        # Add value and RPE, convert to MyTrial
        model_trials = add_value_and_rpe(model_trials, fit_weights(model_trials), 0.93)
        my_trial_dict[model_path] = convert_to_mytrial(model_trials)
        if verbose:
            print("\t\tAdded value and RPE converted to MyTrial.")
    
    return my_trial_dict


def plot_rpes(rpe_groups: Dict[int, RpeGroup]) -> None:
    """
    Plot RPEs for each group.

    Args:
        rpe_groups (Dict[int, RpeGroup]): Dictionary of RpeGroup objects to plot.
    """
    COLORS = [
        "#B19CD9",  # Pastel Purple
        "#C5A3DD",  # Pastel Lilac
        "#D9B7E0",  # Pastel Lavender
        "#ECD1E6",  # Pastel Mauve
        "#FFDCED",  # Pastel Pink
        "#FFB3BA",  # Pastel Light Pink
        "#FFABBB",  # Pastel Rose
        "#EAAECF",  # Pastel Blush
        "#D8A0C4"   # Pastel Orchid
    ]

    last_x, last_y, colors = [], [], []
    for rpe_group in rpe_groups.values():
        color = COLORS[rpe_group.isi_lenght - 6]

        last_x.append(rpe_group.isi_lenght - 1)
        last_y.append(rpe_group.rpes_avg[-1])
        colors.append(color)

        plt.plot(rpe_group.rpes_avg, c=color)

    # plot rpe-max values in scatter plot and reward time as vertical black line
    plt.scatter(last_x, last_y, c=colors)
    plt.plot([0, 0], [-0.05, 0.25], "k--", alpha=0.7)


def plot_compare_rpes(last_vals: Dict[str, List[List[float]]]) -> None:
    """Generates a comparison plot of RPEs for different trials.

    Args:
        last_vals (Dict[str, List[List[float]]]): A dictionary where keys are trial IDs and values are 
            lists of lists containing the last RPE values for each group.

    Returns:
        None
    """
    colors = plt.get_cmap("Pastel2").colors  # Get the pastel color map for plotting

    for idx, (id_, last_vals) in enumerate(last_vals.items()):
        means, errors = [], []
        for outer in last_vals:
            means.append(np.array(outer).mean())
            errors.append(np.array(outer).std())

        # Plot the means with error bars
        plt.errorbar(np.arange(len(means)),  
                     means,                  
                     yerr=errors,            
                     label=id_,              
                     marker="o",             
                     ls="--",                
                     markersize=3,           
                     c=colors[idx % len(colors)],
                     capsize=3)
    
    plt.xlabel("Time of Reward")
    plt.ylabel("RPE")
    plt.title("Comparing RPEs")
    plt.legend()  # Add a legend to the plot


def plot_mses(mses: Dict[str, float], model_confs: Dict[str, Dict[str, float | int]]) -> None:
    """Plots the Mean Squared Errors (MSEs) of different models, distinguishing between Value RNN and DisRNN models.

    Args:
        mses (Dict[str, float]): A dictionary where keys are model paths and values are their respective MSEs.
        model_confs (Dict[str, Dict[str, float | int]]): A dictionary where keys are model paths and values are dictionaries of model configurations.
    """
    def sort_model_key(model_conf: Dict[str, float | int]) -> float:
        return model_conf["hidden_size"] * 100 + int(model_conf["type"] == "vrnn") * 10 + model_conf.get("kl_loss", -1)
    
    # Sort model paths based on the sorting key derived from model configurations
    model_paths_sorted = sorted(mses.keys(), key=lambda path: sort_model_key(model_confs[path]))
    colors = plt.get_cmap("Dark2").colors
    klloss_color_map = [(model_confs[id_].get("kl_loss", 0), None) for id_ in model_paths_sorted][1:]
    klloss_color_map = {kl_loss: colors[kl_loss_idx % len(colors)] for kl_loss_idx, (kl_loss, _) in enumerate(klloss_color_map)}

    def get_correct_colors(model_type: str, kl_loss: float | None = None) -> str | int:
        match model_type:
            case "vrnn":
                return "black"
            case "vrnnuntrained":
                return "red"
            case "disrnn":
                return klloss_color_map[kl_loss]
            case _:
                raise AttributeError(f"Invalid model-type: <{model_type}>")
    
    # Plot each model's MSE with appropriate markers and colors
    for id_ in model_paths_sorted:
        plt.scatter(str(model_confs[id_]['hidden_size']),
                    mses[id_],
                    color=get_correct_colors(model_confs[id_]["type"], model_confs[id_].get("kl_loss", None)),
                    marker="o" if "vrnn" in model_confs[id_]["type"] else "^",
                    alpha=0.6)
        
    # Set y-axis to log scale
    plt.gca().set_yscale('log')
    plt.ylabel("RPE-MSE")
    plt.xlabel("Hidden Size")
    plt.title(f"RPE MSEs - Ommission Probability = {model_confs[model_paths_sorted[0]]['ommision_probs']}")

    # Create a legend for the plot
    legend_elements = ([Line2D([0], [0], marker='o', color='w', label='GRU-RNN', markerfacecolor='black', markersize=10)] +
                       [Line2D([0], [0], marker='o', color='w', label='Untrained GRU-RNN', markerfacecolor='red', markersize=10)] +
                       [Line2D([0], [0], marker='^', color='w', label=f'DisRNN - beta={kl_loss}', markerfacecolor=color, markersize=10) for kl_loss, color in klloss_color_map.items()])
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

#-- Source: The fit_weights and _make_predictions methods are adapted from
#-- https://github.com/mobeets/value-rnn-beliefs

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from starkweather import Trial

import os
import jax
import numpy as np
import matplotlib.pyplot as plt

from custom_datasets import MyStarkweather
from belief_utils import MyTrial, RpeGroup, convert_to_mytrial
from belief_pomdp import add_pomdp_states_beliefs
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


def compare_rpes_plot(last_vals: Dict[str, List[List[float]]]) -> None:
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

    plt.legend()  # Add a legend to the plot


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
        result_mses[key + "_mse"] = mse

    return result_mses


def multi_value_analyze(ommision_probability: float,
                        models_path: str = "data/models",
                        verbose: bool = True) -> Dict[str, List[MyTrial]]:
    """Analyzes multiple value-based RNN models and compares them to a POMDP reference model.

    Args:
        models_path (str, optional): Path to the directory containing model files. Defaults to "data/models".
        verbose (bool, optional): If True, prints detailed logs. Defaults to True.

    Returns:
        Dict[str, List[MyTrial]]: A dictionary where keys are model descriptions and values are lists of MyTrial objects.
    """
    VALUE_RNN_SEARCH = "vrnn"
    VALUE_DISRNN_SEARCH = "disrnn"

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

    # Analyze Value-RNN models
    if verbose:
        print(f"Analyzing Value-RNN models...")
    value_rnn_model_paths: List[str] = list(filter(lambda s: VALUE_RNN_SEARCH in s, os.listdir(models_path)))
    for value_rnn_model_path in value_rnn_model_paths:
        full_path = f"{models_path}/{value_rnn_model_path}"
        lr, hidden_size = value_rnn_model_path.split('_')[1:]
        if verbose:
            print(f"\tFound model at path: {full_path} with conf: learning-rate={lr}, hidden-size={hidden_size}")

        # Initialize model state
        master_key = jax.random.PRNGKey(0)
        value_rnn_state = create_gru_train_state(
            master_key,
            learning_rate=float(lr),
            hidden_size=int(hidden_size),
            batch_size=10,
            seq_length=1_000,
            out_dim=1,
            in_dim=2
        )
        value_rnn_state = load_model_state(path=full_path, state=value_rnn_state)
        if verbose:
            print("\t\tModel state loaded.")

        # Evaluate model
        value_rnn_trials = eval_value_wrapper(eval_dataset, value_rnn_state, 0.05)
        if verbose:
            print("\t\tModel probed.")

        # Add value and RPE, convert to MyTrial
        value_rnn_trials = add_value_and_rpe(value_rnn_trials, fit_weights(value_rnn_trials), 0.93)
        my_trial_dict[f"Value-RNN - {hidden_size}"] = convert_to_mytrial(value_rnn_trials)
        if verbose:
            print("\t\tAdded value and RPE converted to MyTrial.")
    
    return my_trial_dict

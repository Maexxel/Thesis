#-- Source: The
#--     - fit_weights
#--     - _make_predictions 
#--     - safelog
#--     - get_loglike
#-- methods are adapted from https://github.com/mobeets/value-rnn-beliefs

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from starkweather import Trial

import jax
import numpy as np
from scipy.linalg import lstsq
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from custom_datasets import MyStarkweather
from belief_utils import MyTrial, convert_to_mytrial
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


def safelog(x):
    y = x.copy()
    y[y == 0] = np.finfo(np.float32).eps
    return np.log(y)


def get_loglike(zs, ss) -> float:
    """
    Calculates the log-likelihood between a models hidden state (zs)
    and the belief state.
    """
    clf = LogisticRegression(max_iter=int(1e4))
    clf.fit(zs, ss)
    pred1 = softmax(clf.decision_function(zs), axis=-1)
    return np.mean(safelog(pred1[np.arange(len(ss)), ss]))


def get_mean_value(trials):
    """
    Calculates the mean predicted value over all model trials.
    """
    values_normal = {trial.isi: [] for trial in trials}
    max_iti = max([trial.iti for trial in trials])
    for trial in trials:
        padded_ar = np.pad(trial.value, (max_iti - trial.iti, 0), mode="constant")
        padded_ar[padded_ar == 0] = np.nan
        values_normal[trial.isi].append(padded_ar)

    for k in values_normal.keys():
        values_normal[k] = np.nanmean(np.vstack(values_normal[k]), axis=0)

    return values_normal


def get_mean_rpe(trials):
    """
    Calcualtes the mean rpe over all model trials.
    """
    rpes = {trial.isi: [] for trial in trials}

    for trial in trials:
        if trial.reward_size == 1:
            rpes[trial.isi].append(trial.rpe[-1])

    return rpes


def get_r2(hidden_states, pomdp_beliefs) -> float:
    """
    Calculates the correlation r2. If bias should be added, comment in the below line.
    """
    # adding bias or not
    # hidden_states = np.hstack([hidden_states, np.ones((hidden_states.shape[0],1))])
    w = lstsq(hidden_states, pomdp_beliefs)[0]
    
    Yhat = hidden_states @ w
    top = pomdp_beliefs - Yhat
    return 1 - (np.var(top)/np.var(pomdp_beliefs))


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


def fill_mytrial(ommision_probability: float,
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
                batch_size=2,
                seq_length=20_000,
                out_dim=1,
                in_dim=2
            )
        elif model_conf["type"] == DISRNN_ID:
            model_state = create_disrnn_train_state(
                master_rng_key=master_key,
                learning_rate=model_conf["learning_rate"],
                hidden_size=model_conf["hidden_size"],
                batch_size=2,
                seq_length=20_000,
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


def plot_mean_rpes(results):
    RPE_TIMES = list(range(6, 15))
    ID_COLORS = {"disrnn": "green", "vrnn_": "blue", "untrained": "red", "pomdp": "orange"}
    ID_SHAPES = {"disrnn": "^", "vrnn_": "o", "untrained": "o", "pomdp": "o"}
    ID_SIZE = {"disrnn": 30, "vrnn_": 15, "untrained": 20, "pomdp": 15}
    rpes = {id_: {t: [] for t in RPE_TIMES} for id_ in ["disrnn", "vrnn_", "untrained", "pomdp"]}
    
    for model_name, res in results.items():
        for identifier in rpes.keys():
            if identifier not in model_name or "00001" in model_name:
                continue
            for t in rpes[identifier].keys():
                rpes[identifier][t].append(np.mean(res["mean_rpe"][t]))

    plt.figure(figsize=(4, 6))

    for identifier, values in rpes.items():
        for t in RPE_TIMES:
            plt.scatter([t] * len(values[t]),
                        values[t],
                        c=ID_COLORS[identifier],
                        s=3, alpha=.3, marker=ID_SHAPES[identifier])
            plt.scatter(t, np.mean(values[t]),
                        c=ID_COLORS[identifier],
                        marker=ID_SHAPES[identifier],
                        s=ID_SIZE[identifier], alpha=.8)
    
    legend_elements = ([Line2D([], [], marker='^', color='green', label='DisRNNs',  linestyle="None")] +
                       [Line2D([], [], marker='o', color='blue', label='GRU-RNN',  linestyle="None")] +
                       [Line2D([], [], marker='o', color='red', label='Untrained GRU-RNN',  linestyle="None")] +
                       [Line2D([], [], marker='o', color='orange', label='POMDP',  linestyle="None")])
    plt.legend(handles=legend_elements, loc='center')#, bbox_to_anchor=(1, 0.5))
    plt.xticks(np.arange(6, 15), np.arange(6, 15))
    plt.title("RPE - Overview $p_o=0$")
    plt.xlabel("ISI-Time")
    plt.ylabel("RPE Average")


def plot_metrics(mses: Dict[str, float],
              model_confs: Dict[str, Dict[str, float | int]],
              title: str) -> None:
    """Plots the Mean Squared Errors (MSEs) of different models, distinguishing between Value RNN and DisRNN models.

    Args:
        mses (Dict[str, float]): A dictionary where keys are model paths and values are their respective MSEs.
        model_confs (Dict[str, Dict[str, float | int]]): A dictionary where keys are model paths and values are dictionaries of model configurations.
    """
    matplotlib.rcParams.update({'font.size': 15})
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
    if title == 'RPE-MSE':
        plt.gca().set_yscale('log')
    plt.ylabel(title)
    plt.xlabel("Hidden Size")
    title_specific = f" - Ommission Probability = {model_confs[model_paths_sorted[0]]['ommision_probs']}"
    plt.title(f"{title}{title_specific}")

    # Create a legend for the plot
    legend_elements = ([Line2D([0], [0], marker='o', color='w', label='GRU-RNN', markerfacecolor='black', markersize=10)] +
                       [Line2D([0], [0], marker='o', color='w', label='Untrained GRU-RNN', markerfacecolor='red', markersize=10)] +
                       [Line2D([0], [0], marker='^', color='w', label=f'DisRNN - beta={kl_loss}', markerfacecolor=color, markersize=10) for kl_loss, color in klloss_color_map.items()])
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def multi_value_analyze2(ommision_probability: float,
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
                                  len_sequences=20_000,
                                  omission_probability=ommision_probability,
                                  iti_p=1/8,
                                  iti_min=10)

    # Create POMDP reference model
    if verbose:
        print("Creating POMDP reference model...")
    
    result = {}
    raw_pomdp_trials = add_pomdp_states_beliefs(eval_dataset, ommision_probability, 1/8, 10)
    raw_pomdp_trials = add_value_and_rpe(raw_pomdp_trials, fit_weights(raw_pomdp_trials), gamma=0.93)
    pomdp_beliefs = np.vstack([trial.Z for trial in raw_pomdp_trials])

    if ommision_probability > 0.:
        example_trial_idx = -1
        for idx in range(1, len(raw_pomdp_trials) - 1):
            if raw_pomdp_trials[idx].reward_size == 1 and raw_pomdp_trials[idx + 1].reward_size == 0:
                example_trial_idx = idx
                break

    result["pomdp"] = {}
    result["pomdp"]["mean_values"] = get_mean_value(raw_pomdp_trials)
    result["pomdp"]["mean_rpe"] = get_mean_rpe(raw_pomdp_trials)
    pomdp_rpes = np.hstack([trial.rpe for trial in raw_pomdp_trials])
    
    if verbose:
        print(f"Analyzing models...")

    for model_path in model_confs.keys():
        model_conf = model_confs[model_path]

        # Check if model was trained on correct omission probability
        if ommision_probability != model_conf["ommision_probs"]:
            continue
        if verbose:
            print(f"\tFound model at path: {model_path} with conf: {', '.join(key + '=' + str(value) for key, value in model_conf.items())}")

        model_result = {}
        result[model_path] = model_result

        # Initialize model state
        master_key = jax.random.PRNGKey(0)
        if model_conf["type"] == VALUE_ID or model_conf["type"] == "vrnnuntrained":
            model_state = create_gru_train_state(
                master_key,
                learning_rate=model_conf["learning_rate"],
                hidden_size=model_conf["hidden_size"],
                batch_size=2,
                seq_length=20_000,
                out_dim=1,
                in_dim=2
            )
        elif model_conf["type"] == DISRNN_ID:
            model_state = create_disrnn_train_state(
                master_rng_key=master_key,
                learning_rate=model_conf["learning_rate"],
                hidden_size=model_conf["hidden_size"],
                batch_size=10,
                seq_length=20_000,
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
        model_test_trials = eval_value_wrapper(eval_dataset, model_state, sigma=0.05, true_output=1)
        if verbose:
            print("\t\tModel probed.")

        # Add value and RPE, convert to MyTrial
        model_test_trials = add_value_and_rpe(model_test_trials, fit_weights(model_test_trials), 0.93)
        if verbose:
            print("\t\tAdded value and RPEs.")

        model_rpes = np.hstack([trial.rpe for trial in model_test_trials])
        model_result["MSE"] = np.mean(np.square(pomdp_rpes - model_rpes))
        if verbose:
            print("\t\tCalculated MSE: ", model_result["MSE"])

        r2 = get_r2(np.vstack([trial.Z for trial in model_test_trials]),pomdp_beliefs)
        model_result["r2"] = r2
        if verbose:
            print("\t\tCalculated models r2: ", model_result["r2"])

        model_result["ll"] = get_loglike(np.vstack([trial.Z for trial in model_test_trials]),
                                         np.hstack([trial.S for trial in model_test_trials]))
        if verbose:
            print("\t\tCalculated models log-likliehood: ", model_result["ll"])

        model_result["mean_values"] = get_mean_value(model_test_trials)
        if verbose:
            print("\t\tCalulated Mean Values.")

        model_result["mean_rpe"] = get_mean_rpe(model_test_trials)
        if verbose:
            print("\t\tCalulated Mean RPE.")

        if model_conf["hidden_size"] == 10 and ommision_probability > 0:
            plt.figure(figsize=(12, 4))
            z1 = model_test_trials[example_trial_idx].Z.T
            z1 *= 1 / np.max(z1)
            z2 = model_test_trials[example_trial_idx + 1].Z.T
            z2 *= 1 / np.max(z2)
            plt.imshow(np.hstack([z1, z2]), cmap='Oranges')
            xs = np.vstack([model_test_trials[example_trial_idx].X, model_test_trials[example_trial_idx + 1].X])
            plt.xticks(np.where(xs == 1)[0],
                    ["c", "r", "c"])
            plt.yticks(np.arange(model_test_trials[0].Z.shape[1]),
                       np.arange(model_test_trials[0].Z.shape[1]) + 1)
            plt.title("Hidden state activation over time")
            plt.ylabel("Hidden variables")
            plt.xlabel("Time")
            plt.tight_layout()
            plt.show()

    return result

#-- Source: The fit_weights and _make_predictions methods are adapted from
#-- https://github.com/mobeets/value-rnn-beliefs

from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict
if TYPE_CHECKING:
    from starkweather import Trial

import numpy as np
import matplotlib.pyplot as plt

from belief_utils import MyTrial, RpeGroup


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


def calc_rpe_groups(trials: List[MyTrial]) -> Dict[int, RpeGroup]:
    """
    Calculate RPE groups from a list of MyTrial objects.

    Args:
        trials (List[MyTrial]): List of MyTrial objects to process.

    Returns:
        Dict[int, RpeGroup]: Dictionary of RpeGroup objects keyed by ISI length.
    """
    group_dict: Dict[int, RpeGroup] = {}

    for trial in trials:
        if not trial.got_reward:
            continue
        # get current group
        current_rpe_group = group_dict.get(trial.isi_lenght, None)
        if current_rpe_group is None:
           current_rpe_group = RpeGroup(isi_lenght=trial.isi_lenght)
           group_dict[trial.isi_lenght] = current_rpe_group

        # shorten rpe_array and error checking
        shortend_rpes = trial.rpes[trial.iti_lenght:]
        assert trial.isi_lenght == current_rpe_group.isi_lenght, \
            f"Shortened RPEs-Array <{len(shortend_rpes)}> doesn't match ISI-length in trial {trial.idx_in_trial} <{current_rpe_group.isi_lenght}>"
        
        current_rpe_group.rpes.append(shortend_rpes)

    for rpe_group in group_dict.values():
        rpe_group.calc_rpe_avg()
        rpe_group.calc_rpe_max()
    
    return group_dict

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

    max_x, max_y, colors = [], [], []
    for rpe_group in rpe_groups.values():
        color = COLORS[rpe_group.isi_lenght - 6]

        max_x.append(rpe_group.isi_lenght - 1)
        max_y.append(rpe_group.group_max)
        colors.append(color)

        plt.plot(rpe_group.rpes_avg, c=color)

    # plot rpe-max values in scatter plot and reward time as vertical black line
    plt.scatter(max_x, max_y, c=colors)
    plt.plot([0, 0], [-0.05, 0.25], "k--", alpha=0.7)
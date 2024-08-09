#-- Source: All methods are slightly adapted versions of their originals, which can be found at
#-- https://github.com/mobeets/value-rnn-beliefs

from __future__ import annotations
from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from starkweather import Trial

import numpy as np
import scipy
import matplotlib.pyplot as plt
from custom_datasets import MyStarkweather


NULL = 0
STIM = 1
REW = 2


def _transition_distribution(K, reward_times, reward_hazards, p_omission, ITIhazard, iti_times=None):
    """
    T[i,j] = P(s'=j | s=i)
    """
    if iti_times is None:
        ITI_start = -1
        iti_times = []
    else:
        ITI_start = iti_times[0]
        assert(iti_times[-1] == K-1), 'last iti time should be last state'
    T = np.zeros((K,K))
    
    # no probability of transitioning out of isi during this time
    for k in np.arange(min(reward_times)):
        T[k,k+1] = 1.0
    
    # isi
    for t,h in zip(reward_times, reward_hazards):
        T[t,t+1] = 1-h
        T[t,ITI_start] = h
    # T[-2,-1] = 1
    T[reward_times.max(),ITI_start] = 1
    
    # iti
    for t in iti_times[:-1]:
        T[t,t+1] = 1

    # transitions out of last iti:
    T[-1,-1] = 1 - ITIhazard # 1-(ITIhazard*(1-p_omission)) # = 1- (itih - itih*p_omit) = 1 - itih + itih*p_omit
    T[-1,ITI_start] += ITIhazard*p_omission # += so that when ITI_start == -1, we add probs
    T[-1,0] = ITIhazard*(1-p_omission)
    
    return T


def _observation_distribution(K, reward_times, p_omission, ITIhazard, iti_times=None):
    """
    P(x'=m | s=i, s'=j), for m in {NULL, STIM, REW}
    """    
    O = np.zeros((K,K,3))
    if iti_times is None:
        ITI_start = -1
    else:
        ITI_start = iti_times[0]
        assert(iti_times[-1] == K-1), 'last iti time should be last state'
    
    # progressed through time
    for k in np.arange(K-1):
        O[k,k+1,:] = [1,0,0]
    
    # obtained reward
    O[reward_times,ITI_start,:] = [0,0,1]
    
    # stim onset
    O[-1,0,:] = [0,1,0]
    
    # iti
    if np.arange(K)[ITI_start] == K-1:
        O[-1,-1,NULL] = 1-(ITIhazard*p_omission) # stayed in iti
        O[-1,-1,STIM] = ITIhazard*p_omission # omission trial
        O[-1,-1,REW] = 0 # never happens
    else:
        O[-1,-1,:] = [1,0,0] # will always see NULL
        if p_omission > 0:
            O[-1,ITI_start,:] = [0,1,0] # will always see STIM
    return O


def _get_transistion_observation_probs(omission_probability: float,
                                       iti_p: float,
                                       iti_states: int):
    rts = np.arange(1.2, 3.0, 0.2)
    reward_times = (rts/0.2).astype(int)
    ISIpdf = scipy.stats.norm.pdf(rts, rts.mean(), 0.5)
    ISIpdf = ISIpdf/ISIpdf.sum()

    K = reward_times.max() + 1 + iti_states
    ISIcdf = np.cumsum(ISIpdf)
    ISIhazard = ISIpdf
    ISIhazard[1:] = ISIpdf[1:]/(1-ISIcdf[:-1])
    reward_hazards = ISIhazard
    # reward_times = np.round(reward_times / bin_size, 6).astype(int)-1
    iti_times = np.arange(reward_times.max()+1, K)
    
    T = _transition_distribution(K, reward_times, reward_hazards, omission_probability, iti_p, iti_times=iti_times)
    O = _observation_distribution(K, reward_times, omission_probability, iti_p, iti_times=iti_times)
    return T,O


def _get_states_and_observations(trials, cue=0, iti_min=0):
    xs = []
    ss = []
    for i,trial in enumerate(trials):
        if trial.cue != cue:
            continue
        lastTrialWasOmission = trials[i-1].y.sum() == 0
        for t in np.arange(trial.trial_length):            
            if t == trial.iti:
                x = STIM
            elif t == trial.iti + trial.isi:
                x = REW
            else:
                x = NULL
            
            if t >= trial.iti and trial.y.sum() == 0: # omission
                s = np.min([t - trial.iti - iti_min, 0])
                if x == REW:
                    x = NULL
            else:
                if t < trial.iti and lastTrialWasOmission:
                    s = 0
                elif t < trial.iti or t >= (trial.iti + trial.isi):
                    s = np.min([t + 1 - iti_min, 0]) # +1 for an off-by-one fix
                else:
                    s = t - trial.iti + 1
                if x == REW:
                    s = -iti_min # REW means we are in FIRST iti
            xs.append(x)
            ss.append(s)
    S = np.array(ss)
    s_range = S.max() - S.min()
    S[S <= 0] += (1 + s_range) # make ITI last
    S -= 1 # shift so that lowest state is 0

    return S, np.array(xs)


def _get_beliefs(observations, T, O):
    def initial_belief(K, iti_min=0):
        b = np.zeros(K)
        b[-(iti_min+1)] = 1.0 # start knowing we are in ITI
        return b

    b = initial_belief(T.shape[0])
    B = []
    for i,x in enumerate(observations):
        b = b.T @ (T * O[:,:,x])
        b = b/b.sum()
        B.append(b)
    B = np.vstack(B)
    if np.isnan(B).any():
        print("NaN in beliefs. Something went wrong!")
    return B


def add_pomdp_states_beliefs(experiment: MyStarkweather,
                             omission_probability: float,
                             iti_p: float,
                             iti_min: int) -> List[Trial]:
    assert isinstance(experiment, MyStarkweather), "Experiment mus be of type <MyStarkweather>."
    trials: List[Trial] = experiment.get_raw_trials()
    T, O = _get_transistion_observation_probs(omission_probability=omission_probability,
                                              iti_p=iti_p,
                                              iti_states=iti_min + 1)
    S, observations = _get_states_and_observations(trials, cue=0, iti_min=iti_min)
    B = _get_beliefs(observations, T, O)

    i = 0
    for trial in trials:
        trial.Z = B[i:(i+trial.trial_length)]
        trial.S = S[i:(i+trial.trial_length)]
        i += trial.trial_length
    
    return trials


def plot_pomdp_tables(trans_prob, obs_prob) -> None:
    for data, is_t in zip([trans_prob, obs_prob], [True, False]):
        plt.figure(figsize=(10, 10))
        if is_t:
            im = plt.imshow(data, vmin=0, vmax=1, aspect='equal', cmap="Oranges")
        else:
            im = plt.imshow(data)

        ax = plt.gca()

        # Major ticks
        ax.set_xticks(np.arange(0, 25, 1))
        ax.set_yticks(np.arange(0, 25, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(1, 26, 1))
        ax.set_yticklabels(np.arange(1, 26, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, 25, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 25, 1), minor=True)

        # Gridlines based on minor ticks
        gridline_color = "w" if is_t else "#808080"
        ax.grid(which='minor', color=gridline_color, linestyle='-', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)

        # axis labels
        ax.set_ylabel("From state")
        ax.set_xlabel("To state")

        plt.title("Transition probabilities" if is_t else "Observation probabilities")
        plt.savefig(f"figures/{'transition' if is_t else 'observation'}_prob_visualization.png")

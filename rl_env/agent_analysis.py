from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

from .agents import LeakyActor3 

X_LABEL_FONT_SIZE = 9
Y_LABEL_FONT_SIZE = X_LABEL_FONT_SIZE
TITLE_FONT_SIZE = 11

REWARD_COLOR = "#78dfb9"
NOREWARD_COLOR = "#fa6800"

V_COLOR = "#afeff9"
THETA_COLOR = "#adb2fb"

Q_RIGHT_COLOR = "#b491ff"
Q_LEFT_COLOR = "#ff8a59"


def analyze_q_session(choices: Iterable[int],
                      rewards: Iterable[int],
                      qs: Iterable[Iterable[float]],
                      save: str | None = None):
    vals_l = {
        "Left, unrewarded": [[], []],
        "Left, rewarded": [[], []],
        "Right, unrewarded": [[], []],
        "Right, rewarded": [[], []],
    }

    vals_r = {
        "Left, unrewarded": [[], []],
        "Left, rewarded": [[], []],
        "Right, unrewarded": [[], []],
        "Right, rewarded": [[], []],
    }
    
    for idx in range(0,len(qs)-1):
        ql_old, ql, qr_old, qr = qs[idx][0], qs[idx+1][0], qs[idx][1], qs[idx+1][1]
        choice, reward = choices[idx], rewards[idx]
        
        if choice == 0 and reward == 0:
            v_key = "Left, unrewarded"
        if choice == 0 and reward == 1:
            v_key = "Left, rewarded"
        if choice == 1 and reward == 0:
            v_key = "Right, unrewarded"
        if choice == 1 and reward == 1:
            v_key = "Right, rewarded"
        
        vals_l[v_key][0].append(ql_old)
        vals_l[v_key][1].append(ql)

        vals_r[v_key][0].append(qr_old)
        vals_r[v_key][1].append(qr)   
        
    fig, ax = plt.subplots(figsize=(12,7), nrows=2, ncols=4)

    for idx, entry in enumerate(vals_l.items()):
        name, data = entry
        ax[0][idx].plot([0,1],[0,1], c="grey", ls="--", lw="0.8")
        ax[0][idx].set_title(name, fontsize=9)
        ax[0][idx].set_box_aspect(1)
        ax[0][idx].plot(data[0], data[1], c=Q_LEFT_COLOR)
        ax[0][idx].set_xlabel("Q-Left", fontsize=8)
        ax[0][idx].set_ylim(0,1)
        ax[0][idx].set_xlim(0,1)

    for idx, entry in enumerate(vals_r.items()):
        name, data = entry
        ax[1][idx].plot([0,1],[0,1], c="grey", ls="--", lw="0.8")
        ax[1][idx].set_title(name, fontsize=9)
        ax[1][idx].set_box_aspect(1)
        ax[1][idx].plot(data[0], data[1], c=Q_RIGHT_COLOR)
        ax[1][idx].set_xlabel("Q-Right", fontsize=8)
        ax[1][idx].set_ylim(0,1)
        ax[1][idx].set_xlim(0,1)

    ax[0][0].set_ylabel("Q-Left updated")
    ax[1][0].set_ylabel("Q-Right updated")
    fig.tight_layout(h_pad=-5)
    
    fig.suptitle("Q-Learning update rules")

    if save is not None:
        plt.savefig(save)


def plot_q_session(choices: Iterable[int],
                   rewards: Iterable[bool],
                   timeseries: Iterable[Iterable[float]] = None,
                   timeseries_colours: Iterable[str] = [Q_LEFT_COLOR, Q_RIGHT_COLOR],
                   timeseries_labels: Iterable[str] = None,
                   timeseries_name: str = None,
                   save: str | None = None) -> None:
    
    assert len(timeseries) == len(timeseries_colours) == len(timeseries_labels)
    timeseries_len = np.max([len(ts) for ts in timeseries])
    x_axis_steps = np.arange(timeseries_len)
    
    plt.figure(figsize=(12,2))
    
    # choice scatter plot
    plt.scatter(x_axis_steps,
                np.where(choices==0, -0.05, 1.05),
                c=[REWARD_COLOR if c else NOREWARD_COLOR for c in rewards],
                marker='|')
    
    # plot timeseries
    for ts_idx, ts in enumerate(timeseries):
        plt.plot(x_axis_steps,
                ts,
                c=timeseries_colours[ts_idx],
                label=timeseries_labels[ts_idx],
                lw=0.7)
    plt.ylabel(timeseries_name)
    
    # set xlimit and plot bounding lines
    xlims = [-1,timeseries_len]
    plt.plot(xlims, [1,1], c="grey", alpha=0.5, ls="--", lw=1)
    plt.plot(xlims, [0,0], c="grey", alpha=0.5, ls="--", lw=1)
    plt.xlim(xlims)
    
    
    # Legend for Choices
    reward_legend_element = [plt.scatter([],[],marker="|", c="grey")]
    lreward_legend = plt.legend(reward_legend_element,
                                ["Left choices"],
                                loc="upper left",
                                bbox_to_anchor=(1.0, 1.04))
    rreward_legend = plt.legend(reward_legend_element,
                                ["Right choices"],
                                loc="lower left",
                                bbox_to_anchor=(1.0, -.04))

    plt.gca().add_artist(lreward_legend)
    plt.gca().add_artist(rreward_legend)
    
    # legend
    plt.legend(loc="center left",  bbox_to_anchor=(1, 0.5))
    plt.title("Q-Agent example Session")

    if save is not None:
        plt.savefig(save, bbox_inches=Bbox([[.7,-0.2],[12.6,2.1]]))


def plot_leaky_session(choices: Iterable[int],
                       rewards: Iterable[int],
                       v: Iterable[float],
                       theta: Iterable[float],
                       save: str | None = None):
    fig, ax1 = plt.subplots(figsize=(12,2))

    # plot v
    x_axis = np.arange(len(v))
    ax1.plot(x_axis, v, label = "V", c=V_COLOR, lw=1)
    ax1.set_ylabel('V', color=V_COLOR)

    # plot theta (share x-axis)
    ax2 = ax1.twinx()
    ax2.plot(x_axis, theta, label = "Theta", c=THETA_COLOR, lw=1)
    ax2.set_ylabel('Theta', color=THETA_COLOR)
    ax2.set_ylim(-6,6)

    # choice scatter plot
    ax1.scatter(np.arange(len(choices)),
                np.where(choices==0, -0.05, 1.05),
                c=[REWARD_COLOR if c else NOREWARD_COLOR for c in rewards],
                marker='|')

    # Lines Legend (for V and Theta)
    line_legend = [Line2D([0], [0], color=V_COLOR, lw=4),
                   Line2D([0], [0], color=THETA_COLOR, lw=4),]
    ax1.legend(line_legend, ['V', 'Theta'], loc="center left",  bbox_to_anchor=(1.06, 0.5))

    # Legend for Choices
    reward_legend_element = [plt.scatter([],[],marker="|", c="grey")]
    lreward_legend = plt.legend(reward_legend_element,
                                ["Left choices"],
                                loc="upper left",
                                bbox_to_anchor=(1.06, 1.04))
    rreward_legend = plt.legend(reward_legend_element,
                                ["Right choices"],
                                loc="lower left",
                                bbox_to_anchor=(1.06, -.04))

    fig.gca().add_artist(lreward_legend)
    fig.gca().add_artist(rreward_legend)

    # set axis limits, bounding lines and xlabel
    xlims = [-1,len(choices)]
    ax1.plot(xlims, [1,1], c="grey", alpha=0.5, ls="--", lw=1)
    ax1.plot(xlims, [0,0], c="grey", alpha=0.5, ls="--", lw=1)
    ax1.set_xlim([-1,len(choices)])
    ax1.set_xlabel("Trial")
    
    fig.suptitle("Leaky-Actor-Critic example Session")

    if save is not None:
        plt.savefig(save, bbox_inches=Bbox([[.7,-0.2],[12.9,2.1]]))

def analyze_leaky_agent(leaky_agent: LeakyActor3, save: str | None = None) -> None:
    """
    Analyze the leaky compute of a LeakyActor3 agent and plot the results.

    Args:
        leaky_agent (LeakyActor3): The leaky agent to be analyzed.
        save (str | None): Path to save the plot. If None, the plot is not saved.

    Returns:
        None
    """
    titles = ('Left, Unrewarded', 'Left, Rewarded', 'Right, Unrewarded', 'Right, Rewarded')
    observations = ([0, 0], [0, 1], [1, 0], [1, 1])

    lim = 2
    hidden_size = 2
    state_bins = np.linspace(-lim, lim, 20)

    # Prepare input data for the model
    bloated_observations = np.repeat(np.array(observations), len(state_bins) * len(state_bins), axis=0)
    bloated_carries = np.zeros((len(bloated_observations), hidden_size))

    # Set up carries with state bins
    second_latent_input = np.tile(np.repeat(state_bins, len(state_bins)), len(observations))
    bloated_carries[:, 1] = (second_latent_input + 2) / 4
    first_latent_input = np.tile(state_bins, len(state_bins) * len(observations))
    bloated_carries[:, 0] = first_latent_input

    carries = []
    for observation, carrie in zip(bloated_observations, bloated_carries):
        leaky_agent.theta = carrie[0]
        leaky_agent.v = carrie[1]
        leaky_agent.update(choice=observation[0], reward=observation[1])
        carries.append(leaky_agent.q)

    carries = np.array(carries)
    thetas = carries[:, 0].reshape(len(observations), len(state_bins), len(state_bins))
    vs = carries[:, 1].reshape(len(observations), len(state_bins), len(state_bins))

    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1, 1.5])
    colormap = plt.get_cmap('viridis', 20)
    colors = colormap.colors

    # Create subplots as grid
    v_axs = [fig.add_subplot(gs[0, 1]),
             fig.add_subplot(gs[0, 2])]

    t_axes = [fig.add_subplot(gs[1, 0]),
              fig.add_subplot(gs[1, 1]),
              fig.add_subplot(gs[1, 2]),
              fig.add_subplot(gs[1, 3])]

    for ax_idx, title in zip(range(2), ["Unrewarded", "Rewarded"]):
        v_axs[ax_idx].plot(np.linspace(0, 1, 20), vs[:, :, 0][ax_idx], c=V_COLOR)
        v_axs[ax_idx].set_box_aspect(1)
        v_axs[ax_idx].plot([0, 1], [0, 1], c="grey", ls="--", lw="0.8")
        v_axs[ax_idx].set_title(title, fontsize=TITLE_FONT_SIZE)
        v_axs[ax_idx].set_xlabel("V", fontsize=X_LABEL_FONT_SIZE)
        v_axs[ax_idx].set_ylim(0, 1)
        v_axs[ax_idx].set_xlim(0, 1)

    v_axs[0].set_ylabel("Updated V")

    # Plot the output data
    for observation_i in range(len(observations)):
        t_axes[observation_i].plot([-2.1, 2.1], [-2.1, 2.1], c="grey", ls="--", lw="0.8")
        t_axes[observation_i].plot([0, 0], [-2.1, 2.1], c="grey", lw="0.7")
        t_axes[observation_i].plot([-2.1, 2.1], [0, 0], c="grey", lw="0.7")

        for v_idx in range(20):
            t_axes[observation_i].plot(np.linspace(-2, 2, 20), thetas[observation_i][v_idx], color=colors[v_idx])

        t_axes[observation_i].set_title(titles[observation_i], fontsize=TITLE_FONT_SIZE)
        t_axes[observation_i].set_xlim(-lim, lim)
        t_axes[observation_i].set_ylim(-lim, lim)
        t_axes[observation_i].set_xlabel('Theta')
        t_axes[observation_i].set_box_aspect(1)

    cb_ax = fig.add_axes([0.92, 0.08, 0.01, 0.38])
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])

    fig.colorbar(sm, cax=cb_ax, label="V")
    t_axes[0].set_ylabel("Updated Theta")
    fig.suptitle("Leaky-Actor-Critic update rules")

    if save is not None:
        fig.savefig(save, bbox_inches=Bbox([[.8,0],[12,6]]))

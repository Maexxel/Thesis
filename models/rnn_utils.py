from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Dict, Tuple, List, Any, Optional
if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from .disrnn_model import DisRNN
    from .gru_model import SimpleGRU
    from custom_datasets import MyStarkweather
    from torch.utils.data import Dataset
    from starkweather import Trial

import jax
import optax
import os
import time
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax.training import train_state, checkpoints


def value_loss(output: jnp.array, target: jnp.array, gamma: float) -> float:
   V_target = target[:,1:,:] + gamma * output[:,1:,:]
   loss = optax.squared_error(output[:,:-1,:], V_target).mean()
   return loss


def safe_softmax_cross_entropy(logits, labels):
    logsoftmax_x = jax.nn.log_softmax(logits, axis=-1)
    weighted_logsoftmax = jnp.where(
        logits != 0.0, labels * logsoftmax_x, jnp.zeros_like(logsoftmax_x)
    )
    return -jnp.sum(weighted_logsoftmax, axis=-1)


def compute_accuracy(logits: jnp.array, labels: jnp.array) -> Dict[str, float]:
  return jnp.mean(jnp.argmax(logits, -1) == labels.squeeze(-1))


def train_one_epoch(state: train_state.TrainState,
                    dataloader: DataLoader,
                    train_step_fun: Callable[..., Any]) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Train for one epoch on the training set.

    Args:
        state (TrainState): Current training state.
        dataloader (DataLoader): DataLoader providing batches of training data.
        train_step_fun (Callable[..., Any]): Function defining a single training step.

    Returns:
        Tuple[TrainState, Dict[str, float]]: Tuple containing the updated training state and 
            epoch-level metrics averaged across batches.
    """
    from .disrnn_utils import DisRNNTrainState

    state_configured_size = (state.batch_size, state.seq_length, state.in_dim)
    batch_metrics = []

    for batch_x, batch_y in dataloader:
        assert batch_x.shape == state_configured_size, \
            f"XBatch-shape <{batch_x.shape}> doesn't match configured trainstate <{state_configured_size}>."

        # Perform training step
        state, metrics = train_step_fun(state=state, xbatch=batch_x, ybatch=batch_y)
        batch_metrics.append(metrics)

        # RNG handling for DisRNNTrainState
        if isinstance(state, DisRNNTrainState):
            _, new_bottleneck_key = jax.random.split(state.bottleneck_master_key)
            state.replace(bottleneck_master_key=new_bottleneck_key)
            del new_bottleneck_key

    # Convert batch metrics to numpy and compute epoch-level metrics
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    return state, epoch_metrics_np


def _plot_metrics(data):
    """
    Plot epochs vs loss (with a logarithmic y-axis) and a variable number of metrics with stacked y-axes.
    Also plots dotted crosses at the points of lowest values for each metric.

    Parameters:
    data (dict): A dictionary with keys 'loss' and metric names, and their corresponding values as lists or arrays.
                 Example: {'loss': [...], 'accuracy': [...], 'precision': [...], 'recall': [...]}
    """
    epochs = list(range(1, len(data['loss']) + 1))
    loss = data['loss']

    fig, ax1 = plt.subplots()

    # Plot loss with a logarithmic y-axis
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, loss, label='Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Find the epoch with the lowest loss
    min_loss = min(loss)
    min_loss_epoch = epochs[loss.index(min_loss)]

    # Plot dotted cross at the point of lowest loss
    ax1.axvline(x=min_loss_epoch, color='tab:blue', linestyle='dotted', alpha=0.5)
    ax1.axhline(y=min_loss, color='tab:blue', linestyle='dotted', alpha=0.5)

    # Create a secondary y-axis for each metric
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    metric_axes = []
    legend_handles = []

    for i, (metric_name, metric_values) in enumerate(data.items()):
        if metric_name == 'loss':
            continue
        ax = ax1.twinx()
        ax.spines['right'].set_position(('axes', 1 + 0.1 * i))
        ax.plot(epochs, metric_values, label=metric_name, color=colors[i % len(colors)])
        ax.set_ylabel(metric_name, color=colors[i % len(colors)])
        ax.tick_params(axis='y', labelcolor=colors[i % len(colors)])
        metric_axes.append(ax)

        # Find the epoch with the highest value for the current metric
        max_metric_value = max(metric_values)
        max_metric_epoch = epochs[metric_values.index(max_metric_value)]

        # Plot dotted cross at the point of highest metric value
        ax.axvline(x=max_metric_epoch, color=colors[i % len(colors)], linestyle='dotted', alpha=0.5)
        ax.axhline(y=max_metric_value, color=colors[i % len(colors)], linestyle='dotted', alpha=0.5)

        # Add the max value point to the legend
        legend_handles.append(ax.plot([], [], color=colors[i % len(colors)], alpha=0.5,
                                      label=f'{metric_name} (Max: {max_metric_value:.4f}, Epoch: {max_metric_epoch})')[0])

    legend_handles.append(ax1.plot([], [], color='tab:blue', alpha=0.5,
                                   label=f'Loss (Min: {min_loss:.4f}, Epoch: {min_loss_epoch})')[0])

    # Add legend with max values of metrics and min value of loss
    ax1.legend(handles=legend_handles, loc="center")

    # Bring legend to the foreground
    ax1.legend_.set_zorder(100)

    # Adjust layout
    fig.tight_layout()
    
    plt.show()


def train_model(train_state: train_state.TrainState,
                dataloader: DataLoader,
                train_step_fun: Callable[..., Any],
                num_epochs: int = 20,
                print_every_other: int = 1,
                save_path: None | str = None,
                plot_metrics: bool = True) -> Tuple[train_state.TrainState, List[Dict[str, float]]]:
    """
    Train a model over multiple epochs using the provided training state, data loader, and training step function.

    Args:
        train_state (TrainState): Initial training state.
        dataloader (DataLoader): DataLoader providing batches of training data.
        train_step_fun (Callable[..., Any]): Function defining a single training step.
        num_epochs (int, optional): Number of epochs to train the model. Default is 20.
        print_every_other (int, optional): Frequency of epoch printing. Default is 1 (print every epoch).
        save_path (str or None, optional): Path to save checkpoints. Default is None (no checkpoint saving).
        plot_metrics (bool, optional): Whether to plot training metrics after training. Default is True.

    Returns:
        Tuple[TrainState, List[Dict[str, float]]]: Tuple containing the final training state and a list of dictionaries
            containing metrics collected during training for each epoch.
    """
    from .disrnn_utils import DisRNNTrainState

    print(f"Started Training on {', '.join(map(str, jax.devices()))}...\n")
    print(f"  Epoch | Time (s) | Metrics")
    print(f"  ------+----------+----------------->")
    training_metrics: Dict[str, float] = {}

    training_start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        start = time.time()

        # train epoch
        train_state, epoch_metrics = train_one_epoch(state=train_state, dataloader=dataloader, train_step_fun=train_step_fun)

        # rng-handling
        if isinstance(train_state, DisRNNTrainState):
            _, new_bottleneck_key = jax.random.split(train_state.bottleneck_master_key)
            train_state.replace(bottleneck_master_key=new_bottleneck_key)
            del new_bottleneck_key

        # metrics
        for metric_name, metric_value in epoch_metrics.items():
            training_metrics[metric_name] = training_metrics.get(metric_name, []) + [metric_value]

        # output
        if epoch % print_every_other == 0 or epoch == 1:
            time_taken = f"{time.time() - start:.2f}"
            metrics_str = ', '.join([f"{k}: {v:.7f}" for k, v in epoch_metrics.items()])
            print(f"  {str(epoch).rjust(5)} | {time_taken.rjust(8)} | {metrics_str.rjust(10)}")

        # checkpointing
        if save_path is not None:
            try:
                abs_path = os.path.abspath(save_path)
                checkpoints.save_checkpoint(ckpt_dir=abs_path, target=train_state, step=epoch, keep_every_n_steps=10)
            except Exception as e:
                print(f"Saving to disk failed, due to exception: {e}.")

    print(f"  ------+----------+----------------->")
    print(f"\nTraining finished. Trained {epoch} epochs in {time.time() - training_start_time} s.")

    if plot_metrics:
        print("Plotting metrics...")
        _plot_metrics(training_metrics)

    return train_state, training_metrics


def eval(dataset: Dataset,
         model_state: train_state.TrainState,
         inspect: bool = False,
         dataset_start: None | int = None,
         dataset_end: None | int = None,
         true_output: None | int = None,
         verbose: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate a model on a dataset.

    Args:
        dataset (Dataset): Dataset to evaluate the model on.
        model_state (TrainState): Current state of the model.
        inspect (bool, optional): Whether to inspect intermediate results. Default is False.
        dataset_start (int, optional): Start index of dataset for evaluation. Default is None (start from the beginning).
        dataset_end (int, optional): End index of dataset for evaluation. Default is None (evaluate until the end of the dataset).
        true_output (int, optional): Number of true output dimensions to consider. Default is None (use all dimensions).
        verbose (bool, optional): Whether to print verbose output. Default is False.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: If inspect=True, returns a tuple of numpy arrays (carrys, outputs).
            If inspect=False, returns a numpy array of outputs.
    """
    dataset_start = 0 if dataset_start is None else dataset_start
    dataset_end = len(dataset) if dataset_end is None else dataset_end
    assert dataset_start < dataset_end, "dataset_start must be smaller than dataset_end"

    if inspect:
        carrys, outputs = [], []
        for seq in dataset.xs[dataset_start: dataset_end]:
            test_input = np.array(seq)
            carry, output = model_state.apply_fn({'params': model_state.params}, test_input, inspect=True, rngs={"bottleneck_master_key": jax.random.key(0)})
            output = output if true_output is None else output[:,:true_output]

            carrys.append(carry)
            outputs.append(output)

            if verbose:
                print(f"For Input: {test_input.tolist()}, model_predicts: {np.argmax(output, axis=-1)} (raw: {output.tolist()})")
                for idx, c in enumerate(carry):
                    print(f"\tidx {idx}: {c.tolist()}")

        return np.array(carrys), np.array(outputs)

    outputs = []
    for seq in dataset.xs[dataset_start: dataset_end]:
        test_input = np.array([seq])
        output = model_state.apply_fn({'params': model_state.params}, test_input, rngs={"bottleneck_master_key": jax.random.key(0)})
        output = output if true_output is None else output[:,:,:true_output]
        if verbose:
            print(f"For Input: {test_input.tolist()}, model_predicts: {np.argmax(output, axis=-1)} (raw: {output.tolist()})")
        outputs.append(output[0])

    return np.array(outputs)


def eval_value_wrapper(dataset: MyStarkweather,
                       model_state: train_state.TrainState,
                       sigma: float = 0.05,
                       verbose: bool = False) -> List[Trial]:
    """
    Evaluate model outputs on a dataset and map them to trials.

    Args:
        dataset (MyStarkweather): Dataset to evaluate and map to trials.
        model_state (train_state): Current state of the model.
        sigma (float, optional): Standard deviation multiplier for random noise. Default is 0.05.
        verbose (bool, optional): Whether to print verbose output. Default is False.

    Returns:
        List[Trial]: List of trials with model outputs and weights mapped.
    """
    if verbose:
        print("Probing Model (this may take some time)...")
    weights, outputs = model_state.apply_fn({'params': model_state.params},
                                            dataset.xs, inspect=True,
                                            rngs={"bottleneck_master_key": jax.random.key(0)})  # is not uses when using a GRU cell

    # Ensure outputs have expected shape (batch_size/n_seq, seq_len)
    outputs = outputs.squeeze(-1)

    if verbose:
        print("Mapping data to trials...")

    trials: List[Trial] = []
    for exp, exp_weights, exp_outputs in zip(dataset.raw_exps, weights, outputs):
        offset = 0
        for trial in exp.trials:
            upper_offset = offset + trial.trial_length
            if upper_offset > len(exp_outputs):
                break

            # Extract values (outputs) and store them in trial
            trial_values = exp_outputs[offset:upper_offset]
            assert len(trial_values) == trial.trial_length
            trial.value = trial_values

            # Extract weights and store them in trial
            trial_weights = exp_weights[offset:upper_offset]
            assert len(trial_weights) == trial.trial_length
            trial.Z = trial_weights

            trials.append(trial)
            offset += trial.trial_length

    # Add noise to weights based on sigma and standard deviation
    sds = np.vstack([trial.Z for trial in trials]).std(axis=0)
    for trial in trials:
        trial.Z = trial.Z.copy() + sigma * sds * np.random.randn(*trial.Z.shape)

    return trials


def load_model_state(state: train_state.TrainState, path: str, step: int | None = None) -> train_state.TrainState:
    try:
        abs_path = os.path.abspath(path)
        return checkpoints.restore_checkpoint(ckpt_dir=abs_path, target=state, step=step)
    except Exception as e:
        print(f"Saving to disk failed, due to exception: {e}.")


def print_model_param_overview(model: DisRNN | SimpleGRU, test_input: jnp.array) -> None:
    print(model.tabulate(jax.random.key(0),
                         test_input,
                         compute_flops=True,
                         compute_vjp_flops=True,
                         console_kwargs={"width": 200}))
    

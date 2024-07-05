import jax
import optax
import jax.numpy as jnp
from flax.training import train_state

from .rnn_utils import safe_softmax_cross_entropy, compute_accuracy
from .gru_model import SimpleGRU

@jax.jit
def gru_train_step(state: train_state.TrainState, xbatch, ybatch,):
    def loss_fun(params):
        logits = state.apply_fn({'params': params},
                                xbatch)
        loss =  jnp.mean(safe_softmax_cross_entropy(logits, jax.nn.one_hot(ybatch.squeeze(-1), num_classes=2)))
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fun, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "accuracy": compute_accuracy(logits, ybatch)}

class GRUTrainState(train_state.TrainState):
    batch_size: int
    seq_length: int
    in_dim: int

def create_gru_train_state(rng,
                           learning_rate: float,
                           hidden_size: int,
                           in_dim: int,
                           out_dim: int,
                           batch_size: int,
                           seq_length: int) -> GRUTrainState:
    """Creates initial `TrainState`. Also returns model."""
    model = SimpleGRU(hidden_size, out_dim)
    x_train_expample = jnp.ones([batch_size, seq_length, in_dim])
    params = model.init(rng, x_train_expample)['params']
    tx = optax.adamw(learning_rate)
    return GRUTrainState.create(apply_fn=model.apply,
                                params=params,
                                tx=tx,
                                batch_size=batch_size,
                                seq_length=seq_length,
                                in_dim=in_dim)
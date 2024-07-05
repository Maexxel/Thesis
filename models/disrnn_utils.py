from typing import Sequence
import jax
import optax
import jax.numpy as jnp
from flax.training import train_state

from .rnn_utils import safe_softmax_cross_entropy, compute_accuracy
from .disrnn_model import DisRNN

PRNGKey = jax.Array


def disrnn_loss(logits, labels, kl_loss_factor: float):
    true_logits = logits[:,:,:-1]
    kl_loss = jnp.mean(logits[:,:,-1])

    mean_softmax_cross_entropy = jnp.mean(safe_softmax_cross_entropy(
        logits=true_logits, 
        labels=jax.nn.one_hot(labels.squeeze(-1), num_classes=2)))
    return mean_softmax_cross_entropy + kl_loss_factor * kl_loss


@jax.jit
def disrnn_trainstep(state: 'DisRNNTrainState', xbatch, ybatch):
   def loss_fun(params):
      bottleneck_key, new_bottleneck_key = jax.random.split(state.bottleneck_master_key)
      logits = state.apply_fn({'params': params}, xbatch,
                              rngs={"bottleneck_master_key": bottleneck_key})
      del bottleneck_key

      state.replace(bottleneck_master_key=new_bottleneck_key)
      del new_bottleneck_key

      loss = disrnn_loss(logits, ybatch, kl_loss_factor=state.kl_loss_factor)
      return loss, logits
   
   grad_fn = jax.value_and_grad(loss_fun, has_aux=True)
   (loss, logits), grads = grad_fn(state.params)
   state = state.apply_gradients(grads=grads)

   metrics = {"loss": loss, "accuracy": compute_accuracy(logits[:, :, :-1], ybatch)}
   return state, metrics


class DisRNNTrainState(train_state.TrainState):
    true_out_dim: int
    batch_size: int
    seq_length: int
    in_dim: int

    kl_loss_factor: float
    bottleneck_master_key: PRNGKey

def create_disrnn_train_state(master_rng_key: PRNGKey,
                              learning_rate: float,
                              hidden_size: int,
                              batch_size: int,
                              seq_length: int,

                              in_dim: int,
                              out_dim: int,
                              update_mlp_shape: Sequence[int],
                              choice_mlp_shape: Sequence[int],
                              kl_loss_factor: float
                              ) -> DisRNNTrainState:
    
    carry_init_key, param_key, bottleneck_master_key = jax.random.split(master_rng_key, 3)
    model = DisRNN(hidden_size=hidden_size,
                   in_dim=in_dim,
                   out_dim=out_dim,
                   update_mlp_shape=update_mlp_shape,
                   choice_mlp_shape=choice_mlp_shape,
                   carry_init_key=carry_init_key)

    x_train_expample = jnp.ones([batch_size, seq_length, in_dim])
    params = model.init(param_key, x_train_expample)['params']
    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     optax.adam(learning_rate)
                     )    
    return DisRNNTrainState.create(apply_fn=model.apply,
                                   params=params,
                                   tx=tx,
                                   true_out_dim=out_dim,
                                   batch_size=batch_size,
                                   seq_length=seq_length,
                                   in_dim=in_dim,
                                   kl_loss_factor=kl_loss_factor,
                                   bottleneck_master_key=bottleneck_master_key)
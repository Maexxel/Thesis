import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Callable, Sequence, Union, Tuple, Dict, Any

PRNGKey = jax.Array
Dtype = Union[jax.typing.DTypeLike, Any]
Initializer = Union[jax.nn.initializers.Initializer, Callable[..., Any]]

class MLP(nn.Module):
    hidden_neurons: Sequence[int]
    kernel_init: Callable = nn.initializers.lecun_normal()
    activation: Callable = nn.sigmoid
    
    @nn.compact
    def __call__(self, x):
        for k, neurons in enumerate(self.hidden_neurons):
            x = nn.Dense(neurons, kernel_init=self.kernel_init)(x)
            if k != len(self.hidden_neurons) - 1:
                x = self.activation(x)
        return x

def kl_gaussian_loss(mean: jnp.ndarray, var: jnp.ndarray) -> jnp.ndarray:
  r"""
  Src: https://github.com/kstach01/CogModelingRNNsTutorial.git

  Calculate KL divergence between given and standard gaussian distributions.

  KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
          = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
          = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
  Args:
    mean: mean vector of the first distribution
    var: diagonal vector of covariance matrix of the first distribution

  Returns:
    A scalar representing KL divergence of the two Gaussian distributions.
  """
  # jax.debug.# print("Mean-Mu: {x}, Mean-Sigma: {y}", x=mean, y=var)
  return 0.5 * jnp.sum(-jnp.log(var) - 1.0 + var + jnp.square(mean), axis=-1)

   
class DisRNNCell(nn.RNNCellBase):
    hidden_size: int
    in_dim: int
    out_dim: int
    update_mlp_shape: Sequence[int]
    choice_mlp_shape: Sequence[int]

    inspect: bool = False
    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, carry, inputs):
        # validate carry and input
        assert carry.shape == (self.hidden_size, ), f"Carry of DisRNNCell <{carry.shape}> has invalid shape/size, should be <{(self.hidden_size, )}>."
        assert inputs.shape == (self.in_dim, ), f"Input of DisRNNCell <{inputs.shape}> has invalid shape/size, should be <{(self.in_dim, )}>."

        # create needed random splits
        latent_normal_key, update_normal_key = jax.random.split(self.make_rng("bottleneck_master_key"), 2)

        # Update Bottlenecks
        n_update_bootlenecks = (self.hidden_size + self.in_dim) * self.hidden_size
        update_bottleneck_mus = self.param("update_bottleneck_mus",
                                           lambda key, shape: jax.nn.initializers.uniform(0.1)(key, shape) + 1,
                                           (n_update_bootlenecks,))
        update_bottleneck_sigmas = self.param("update_bottleneck_sigmas",
                                              lambda key, shape: jnp.abs(jax.nn.initializers.uniform(0.1)(key, shape)) - 3,
                                           (n_update_bootlenecks,))
        update_bottleneck_sigmas = 2 * jax.nn.sigmoid(update_bottleneck_sigmas)
        
        # duplicate input (-> update_bottleneck for every Update-MLP)
        complete_input = jnp.tile(jnp.hstack([carry, inputs]), self.hidden_size)
        assert complete_input.shape == ((self.hidden_size + self.in_dim) * self.hidden_size, )

        bl_complete_input = update_bottleneck_mus * complete_input + update_bottleneck_sigmas * jax.random.normal(update_normal_key, (n_update_bootlenecks, ))
        del update_normal_key

        # Update MLPs
        for latent_idx in range(self.hidden_size):
            # extraxt input for MLP
            start_idx = latent_idx * (self.hidden_size + self.in_dim)
            end_idx = start_idx + (self.hidden_size + self.in_dim)
            update_mlp_in = bl_complete_input[start_idx:end_idx]
            assert update_mlp_in.shape == (self.hidden_size + self.in_dim, )

            # Calc weight and update (Output of Update-MLP)
            update_mlp_out_raw = MLP(hidden_neurons=self.update_mlp_shape,
                                     activation=nn.relu, name=f"update_mlp_{latent_idx}")(update_mlp_in)
            weight, update = nn.Dense(2)(update_mlp_out_raw)
            weight = jax.nn.sigmoid(weight)

            # Set new carry value
            new_carry_val = (1 - weight) * carry[latent_idx] + weight * update
            carry = carry.at[latent_idx].set(new_carry_val)            

        # Latent Bottlenecks (only sigmas)
        n_latent_bottlenecks = self.hidden_size
        latent_bottleneck_sigmas = self.param("latent_bottleneck_sigmas",
                                              lambda key, shape: jnp.abs(jax.nn.initializers.uniform(0.1)(key, shape))  - 3,
                                           (n_latent_bottlenecks,))
        latent_bottleneck_sigmas = 2 * jax.nn.sigmoid(latent_bottleneck_sigmas)

        bl_carry = carry + latent_bottleneck_sigmas * jax.random.normal(latent_normal_key, (n_latent_bottlenecks, ))
        del latent_normal_key

        # Choice MLP
        output_raw = MLP(self.choice_mlp_shape, activation=nn.relu, name="choice_mlp")(bl_carry)
        output = nn.Dense(self.out_dim)(output_raw)

        kl_loss = jnp.array([kl_gaussian_loss(jnp.hstack([update_bottleneck_mus, carry]),
                                              jnp.hstack([update_bottleneck_sigmas, latent_bottleneck_sigmas]))]
                                              )
        out = jnp.hstack([output, kl_loss])

        if self.inspect:
            return bl_carry, jnp.hstack([bl_carry, out])
        return bl_carry , out
    
    @nn.nowrap
    def initialize_carry(self, rng: PRNGKey, input_shape: Tuple[int, ...]):
        """Initialize the RNN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the cell.

        Returns:
        An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_size,)
        return self.carry_init(rng, mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1


class DisRNN(nn.Module):
    hidden_size: int
    in_dim: int
    out_dim: int
    update_mlp_shape: Sequence[int]
    choice_mlp_shape: Sequence[int]

    carry_init_key: PRNGKey

    def __get_self_args(self) -> Dict[str, Any]:
        return{"hidden_size": self.hidden_size,
               "in_dim": self.in_dim,
               "out_dim": self.out_dim,
               "update_mlp_shape": self.update_mlp_shape,
               "choice_mlp_shape": self.choice_mlp_shape,}

    @nn.compact
    def __call__(self, inputs, inspect=False):
        carry = self.initialize_carry(inputs[:, 0].shape)
        # parallize over batches
        batch_disrnn = nn.vmap(DisRNNCell,
                               in_axes=0,
                               out_axes=0,
                               variable_axes={'params': None},
                               split_rngs={'params': False, "bottleneck_master_key": True})

        # parallize over steps ("fast for-loop")
        scan_disrnn = nn.scan(batch_disrnn,
                              variable_broadcast='params',
                              in_axes=1,
                              out_axes=1,
                              split_rngs={'params': False, "bottleneck_master_key": True},
                              )

        _, out = scan_disrnn(**self.__get_self_args(), inspect=inspect, name="DisRNNCell0")(carry, inputs)
        if inspect:
            return out[:, :, :self.hidden_size], out[:, :, self.hidden_size:]
        return out

    def initialize_carry(self, input_shape):
        # Use fixed random key since default state init fn is just zeros.
        return DisRNNCell(**self.__get_self_args(), inspect=False, parent=None).initialize_carry(
            self.carry_init_key, input_shape
        )  

if __name__ == "__main__":
    # The following code is only used for testing and has no specific meaning

    master_key = jax.random.key(0)
    carry_init_key, param_key, bottleneck_master_key = jax.random.split(master_key, 3) 

    input = jnp.array([[[1., 1.], [1., 1.], [1., 1.]],
                    [[1., 1.], [1., 1.], [1., 1.]]])

    model = DisRNN(hidden_size=5,
                    in_dim=2,
                    out_dim=1,
                    update_mlp_shape=[5, 5, 5],
                    choice_mlp_shape=[2, 2],
                    carry_init_key=carry_init_key)
    del carry_init_key

    params = model.init(param_key, input)["params"]
    del param_key

    print("Output:")
    print(model.apply({"params": params}, input, True, rngs={"bottleneck_master_key": bottleneck_master_key}))
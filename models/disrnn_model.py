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

   
class DisRNNCell(nn.RNNCellBase):
    hidden_size: int
    in_dim: int
    out_dim: int
    update_mlp_shape: Sequence[int]
    choice_mlp_shape: Sequence[int]
    rng_key: PRNGKey

    param_dtype: Dtype = jnp.float32
    carry_init: Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, carry, inputs):
        # validate carry and input
        assert carry.shape == (self.hidden_size, ), f"Carry of DisRNNCell <{carry.shape}> has invalid shape/size, should be <{(self.hidden_size, )}>."
        assert inputs.shape == (self.in_dim, ), f"Input of DisRNNCell <{inputs.shape}> has invalid shape/size, should be <{(self.in_dim, )}>."

        # create needed random splits
        latent_normal, update_normal = jax.random.split(self.rng_key, 2)

        # Update Bottlenecks
        n_update_bootlenecks = self.hidden_size + self.in_dim
        update_bottleneck_mus = self.param("update_bottleneck_mus",
                                           lambda key, shape: jax.nn.initializers.uniform(0.1)(key, shape) + 1,
                                           (n_update_bootlenecks,))
        update_bottleneck_sigmas = self.param("update_bottleneck_sigmas",
                                              lambda key, shape: jax.nn.initializers.uniform(0.1)(key, shape),
                                           (n_update_bootlenecks,))
        
        complete_input = jnp.hstack([carry, inputs])
        bl_complete_input = update_bottleneck_mus * complete_input + update_bottleneck_sigmas * jax.random.normal(update_normal, (n_update_bootlenecks, ))
        del update_normal
        
        # Update MLPs
        for latent_idx in range(self.hidden_size):
            update_mlp_out_raw = MLP(hidden_neurons=self.update_mlp_shape,
                                     activation=nn.relu, name=f"update_mlp_{latent_idx}")(bl_complete_input)
            weight, update = nn.Dense(2)(update_mlp_out_raw)
            new_carry_val = (1 - weight) * carry[latent_idx] + weight * update
            carry = carry.at[latent_idx].set(new_carry_val)            
            
        # Latent Bottlenecks
        n_latent_bottlenecks = self.hidden_size
        latent_bottleneck_mus = self.param("latent_bottleneck_mus",
                                           lambda key, shape: jax.nn.initializers.uniform(0.1)(key, shape) + 1,
                                           (n_latent_bottlenecks,))
        latent_bottleneck_sigmas = self.param("latent_bottleneck_sigmas",
                                              lambda key, shape: jax.nn.initializers.uniform(0.1)(key, shape),
                                           (n_latent_bottlenecks,))

        bl_carry = latent_bottleneck_mus * carry + latent_bottleneck_sigmas * jax.random.normal(latent_normal, (n_latent_bottlenecks, ))
        del latent_normal

        # Choice MLP
        output_raw = MLP(self.choice_mlp_shape, activation=nn.relu, name="choice_mlp")(bl_carry)
        output = nn.Dense(self.out_dim)(output_raw)

        return bl_carry , output # new carry / output
    
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

    rng_key: PRNGKey = jax.random.key(0)

    def __get_self_args(self) -> Dict[str, Any]:
        return{"hidden_size": self.hidden_size,
               "in_dim": self.in_dim,
               "out_dim": self.out_dim,
               "update_mlp_shape": self.update_mlp_shape,
               "choice_mlp_shape": self.choice_mlp_shape,
               "rng_key": self.rng_key}

    @nn.compact
    def __call__(self, inputs, inspect=False):       
        carry = DisRNNCell(**self.__get_self_args()).initialize_carry(self.rng_key, inputs[:, 0].shape)

        if inspect:
            assert len(inputs.shape) == 2, \
                f"Input in Inspect mode must be a single Sequence (no batching), not shape <{inputs.shape}>."

            carry_complete = jnp.zeros((len(inputs), self.hidden_size))
            output_complete = jnp.zeros((len(inputs), self.out_dim))

            disrnn_cell = DisRNNCell(**self.__get_self_args(), name="DisRNNCell0")

            # iterate over every element of the sequence
            for idx, single_x in enumerate(inputs):
                carry, out = disrnn_cell(carry, single_x)

                # store weights (a.k.a carry and output)
                carry_complete = carry_complete.at[idx].set(carry)
                output_complete = output_complete.at[idx].set(out)

            return carry_complete, output_complete

        # parallize over batches
        batch_disrnn = nn.vmap(DisRNNCell,
                               in_axes=0,
                               out_axes=0,
                               variable_axes={'params': None},
                               split_rngs={'params': False})

        # parallize over steps ("fast for-loop")
        scan_disrnn = nn.scan(batch_disrnn,
                              variable_broadcast='params',
                              in_axes=1,
                              out_axes=1,
                              split_rngs={'params': False},
                              )

        return scan_disrnn(**self.__get_self_args(), name="DisRNNCell0")(carry, inputs)[1]

    def initialize_carry(self, input_shape):
        # Use fixed random key since default state init fn is just zeros.
        return DisRNNCell(**self.__get_self_args(), parent=None).initialize_carry(
            self.rng_key, input_shape
        )
        

if __name__ == "__main__":
    test_input = jnp.array([[[1., 2], [2., 2], [3., 2]],
                            [[4., 2], [5., 2], [6., 2]]])

    model = DisRNN(hidden_size=5,
                   in_dim=2,
                   out_dim=2,
                   update_mlp_shape=[5, 5, 5],
                   choice_mlp_shape=[2, 2],
                   rng_key=jax.random.key(0))
    params = model.init(jax.random.key(0), test_input)["params"]

    print("Output:")
    print(model.apply({"params": params}, test_input))

    print(model.tabulate(jax.random.key(0),
                         test_input,
                         compute_flops=True,
                         compute_vjp_flops=True,
                         console_kwargs={"width": 200}))
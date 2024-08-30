import jax
from jax import numpy as jnp
from flax import linen as nn
import numpy as np

class MyGRUCell(nn.Module):
    hidden_size: int
    out_dim: int
    inspect: bool

    @nn.compact
    def __call__(self, carry, x):
        carry, cell_out = nn.GRUCell(self.hidden_size, name="GRUCell0")(carry, x)
        out = nn.Dense(self.out_dim, use_bias=True)(cell_out)
        if self.inspect:
            return carry, jnp.hstack([carry, out])
        return carry, out


class SimpleGRU(nn.Module):
    """A simple unidirectional RNN."""

    hidden_size: int
    out_dim: int  

    @nn.compact
    def __call__(self, x, inspect=False):
        carry = self.initialize_carry(x[:, 0].shape)

        batch_gru = nn.vmap(MyGRUCell,
                            in_axes=0,
                            out_axes=0,
                            variable_axes={'params': None},
                            split_rngs={'params': False})

        scan_gru = nn.scan(batch_gru,
                            variable_broadcast='params',
                            in_axes=1,
                            out_axes=1,
                            split_rngs={'params': False})

        _, out = scan_gru(hidden_size=self.hidden_size,
                            out_dim=self.out_dim, inspect=inspect)(carry, x)
    
        if inspect:
            return out[:, :, :self.hidden_size], out[:, :, self.hidden_size:]
        return out

    def initialize_carry(self, input_shape):
    # Use fixed random key since default state init fn is just zeros.
        return nn.GRUCell(self.hidden_size, parent=None).initialize_carry(
            jax.random.key(0), input_shape
        )
  
def __test_gru_returns_correct_output_shape():
    """
    Src: Adapted from https://github.com/google/flax.git/examples/sst2/models_test.py
    Tests if the simple GRU returns the correct shape.
    """
    batch_size = 2
    seq_len = 3
    embedding_size = 4
    hidden_size = 5

    model = SimpleGRU(hidden_size, embedding_size)
    rng = jax.random.key(0)
    inputs = np.random.RandomState(0).normal(
        size=[batch_size, seq_len, embedding_size]
    )
    # initial_state = model.initialize_carry(inputs[:, 0].shape)
    output, _ = model.init_with_output(rng, inputs)
    assert (batch_size, seq_len, embedding_size) == output.shape


if __name__ == "__main__":
    # The following code is only used for testing and has no specific meaning

    __test_gru_returns_correct_output_shape()

    gru_model = SimpleGRU(5, 2)
    test_input = np.random.RandomState(0).normal(
            size=[5, 4, 3]
        )

    params = gru_model.init(jax.random.key(0), test_input)["params"]
    print(gru_model.tabulate(jax.random.key(1),
                            test_input,
                            compute_flops=True, compute_vjp_flops=True, console_kwargs={"width": 200}))
import jax
from jax import numpy as jnp
from flax import linen as nn
import numpy as np


class SimpleGRU(nn.Module):
  """A simple unidirectional RNN."""

  hidden_size: int
  out_dim: int  

  @nn.compact
  def __call__(self, x, inspect=False):
    carry = self.initialize_carry(x[:, 0].shape)

    # Track all weights and return all weights aswell as the outputs
    if inspect:
      assert len(x.shape) == 2

      carry_complete = jnp.zeros((len(x), self.hidden_size))
      output_complete = jnp.zeros((len(x), self.out_dim))

      cell = nn.GRUCell(self.hidden_size, name="GRUCell0")
      dense = nn.Dense(self.out_dim, use_bias=True)

      # iterate over every element of the sequence
      for idx, single_x in enumerate(x):
        carry, cell_out = cell(carry, single_x)
        out = dense(cell_out)

        # store weights (a.k.a carry and output)
        carry_complete = carry_complete.at[idx].set(carry[0])
        output_complete = output_complete.at[idx].set(out[0])

      return carry_complete, output_complete

    # If not inspect use scan to make training faster
    cell = nn.scan(nn.GRUCell,
                   variable_broadcast='params',
                   in_axes=1,
                   out_axes=1,
                   split_rngs={'params': False},
                   )
    _, cell_out = cell(self.hidden_size, name="GRUCell0")(carry, x)
    return nn.Dense(self.out_dim, use_bias=True)(cell_out)

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
    __test_gru_returns_correct_output_shape()

    gru_model = SimpleGRU(10, 2)
    test_input = np.random.RandomState(0).normal(
            size=[5, 4, 3]
        )

    print(gru_model.tabulate(jax.random.key(1),
                            test_input,
                            compute_flops=True, compute_vjp_flops=True, console_kwargs={"width": 200}))
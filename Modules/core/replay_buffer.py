import jax.numpy as jnp
from jax import random, jit, vmap
from jax.tree_util import tree_map
from functools import partial


class UniformReplayBuffer():
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        num_qubits: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_qubits = num_qubits

    def empty_buffer(
        self,
    ):
        buffer_state = {}

        buffer_state["states"] = jnp.zeros(
            shape=(self.buffer_size, self.num_qubits), dtype=jnp.int32)
        buffer_state["actions"] = jnp.zeros(
            shape=(self.buffer_size), dtype=jnp.int32)
        buffer_state["rewards"] = jnp.zeros(
            shape=(self.buffer_size), dtype=jnp.float32)
        buffer_state["scores"] = jnp.zeros(
            shape=(self.buffer_size), dtype=jnp.float32)
        buffer_state["next_states"] = jnp.zeros(
            shape=(self.buffer_size, self.num_qubits), dtype=jnp.int32)
        buffer_state["dones"] = jnp.zeros(
            shape=(self.buffer_size), dtype=jnp.bool)

        return buffer_state

    @partial(jit, static_argnums=(0))
    def add(
        self,
        buffer_state: dict[str, jnp.ndarray],
        experience: tuple,
        idx: int,
    ):
        """
        experience: tuple of (state, score, action, reward, next_state, done)
        """
        state, score, action, reward, next_state, done = experience
        idx = idx % self.buffer_size

        buffer_state["states"] = buffer_state["states"].at[idx].set(state)
        buffer_state["scores"] = buffer_state["scores"].at[idx].set(score)
        buffer_state["actions"] = buffer_state["actions"].at[idx].set(action)
        buffer_state["rewards"] = buffer_state["rewards"].at[idx].set(reward)
        buffer_state["next_states"] = buffer_state["next_states"].at[idx].set(
            next_state)
        buffer_state["dones"] = buffer_state["dones"].at[idx].set(done)

        return buffer_state

    @partial(jit, static_argnums=(0))
    def sample(
        self,
        key: random.PRNGKey,
        buffer_state: dict,
        current_buffer_size: int,
    ):

        @partial(vmap, in_axes=(0, None))  # iterate over the indexes
        def sample_batch(indexes, buffer):
            """
            For a given index, extracts all the values from the buffer
            """
            # Equivilent to this but better: {key: val[indexes] for key, val in buffer.items()}
            return tree_map(lambda x: x[indexes], buffer)

        key, subkey = random.split(key)
        indexes = random.randint(
            subkey,
            shape=(self.batch_size,),
            minval=0,
            maxval=current_buffer_size,
        )
        experiences = sample_batch(indexes, buffer_state)

        return experiences, subkey

    # @partial(jit, static_argnums=(0))
    # def get_score(self, buffer_state: dict[str, jnp.ndarray], state: jnp.ndarray):
    #     """
    #     Given a state, returns its corresponding score if it exists in the buffer.
    #     If the state is not found, returns NaN.
    #     """
    #     # Check which entries in buffer_state["states"] match the given state
    #     matches = jnp.all(buffer_state["states"] == state, axis=1)

    #     # Find the index where the state exists (if any)
    #     matching_indices = jnp.where(matches, size=1, fill_value=-1)[0]

    #     # If no match is found, return NaN
    #     score = jnp.where(matching_indices == -1, jnp.nan,
    #                       buffer_state["score"][matching_indices])

    #     return score

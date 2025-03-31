
from jax import vmap
import jax.numpy as jnp
from quantum_error_correction_code import QEC


class PMLD:

    def __init__(
        self,
        code: QEC,
        error_prob: jnp.ndarray,
        parity_info: tuple[jnp.ndarray],
    ):
        """
        code: The quantum error correction code that should be decoded

        error_prob: An [X, Y, Z] array describing the error rate for each Pauli-error
        """
        # Number of data qubits in the code
        n = code.hx_original.shape[1]

        # Generate a complete list of all possible error the code can have
        # NOTE: This gets exponentially hard for bigger codes
        all_error_idxs = jnp.zeros(
            shape=(2**(2*n), 2*n),
            dtype=jnp.int32
        ) + jnp.arange(0, 2**(2*n))[:, None]
        all_errors = (all_error_idxs // 2**jnp.arange(0, 2*n)) % 2

        # Calculate the syndromes and logicals for each possible error
        all_syndromes, all_logicals = vmap(
            code.syndrome,
            in_axes=(0, None),
            out_axes=0
        )(all_errors.reshape((-1, 2, n)), parity_info)
        all_syndrome_idxs = jnp.dot(
            all_syndromes, 2**jnp.arange(all_syndromes.shape[1]))
        all_logical_idxs = jnp.dot(
            all_logicals, 2**jnp.arange(all_logicals.shape[1]))

        # Calculates the likelihood for each error
        probs = jnp.array([
            1-error_prob.sum(),  # I
            error_prob[0],  # Z
            error_prob[2],  # Y
            error_prob[1],  # Z
        ])
        likelihood = jnp.prod(
            probs[all_errors[:, :n] + 2*all_errors[:, n:]],
            axis=1
        )

        # Add up the likelihoods in a table based on their syndrome and logicals
        self.likelihood_table = jnp.zeros(shape=(
            2**all_syndromes.shape[1],
            2**all_logicals.shape[1],
        )).at[(all_syndrome_idxs, all_logical_idxs)].add(likelihood)

    def decode(
        self,
        syndrome: jnp.ndarray,
    ):
        syndrome_idx = jnp.dot(syndrome, 2**jnp.arange(syndrome.shape[0]))
        return jnp.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ])[self.likelihood_table[syndrome_idx].argmax()]

    def decode_batch(
        self,
        syndromes: jnp.ndarray,
    ):
        return vmap(self.decode)(syndromes)

    def exact_logical_error_rate(
        self,
    ):
        return 1 - self.likelihood_table.max(axis=1).sum()

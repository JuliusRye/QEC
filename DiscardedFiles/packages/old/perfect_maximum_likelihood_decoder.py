from jax import vmap
import jax.numpy as jnp


class PMLD:

    def __init__(
        self,
        H_syndrome: dict,
        H_logicals: dict,
    ):
        """Perfect SC3 decoder + lowest possible SC3 logical error rates"""
        def _syndrome_logicals(
            error,
        ):
            syndrome = (
                jnp.matmul(H_syndrome['x'], error == 1,
                           preferred_element_type=jnp.int32) +
                jnp.matmul(H_syndrome['y'], error == 2,
                           preferred_element_type=jnp.int32) +
                jnp.matmul(H_syndrome['z'], error == 3,
                           preferred_element_type=jnp.int32)
            ) % 2
            logicals = (
                jnp.matmul(H_logicals['x'], error == 1,
                           preferred_element_type=jnp.int32) +
                jnp.matmul(H_logicals['y'], error == 2,
                           preferred_element_type=jnp.int32) +
                jnp.matmul(H_logicals['z'], error == 3,
                           preferred_element_type=jnp.int32)
            ) % 2
            return syndrome, logicals

        def _error_to_prob(
            probability: jnp.ndarray,
            qubit_errors: jnp.ndarray,
        ):
            return probability[qubit_errors]

        self._error_to_prob_batch = vmap(
            _error_to_prob,
            in_axes=[0, 1],
            out_axes=1
        )

        all_errors_idx = jnp.zeros(
            shape=(4**9, 9), dtype=jnp.int32) + jnp.arange(0, 4**9)[:, None]
        self.all_errors = (all_errors_idx // 4**jnp.arange(0, 9)) % 4
        batch_syndrome_logicals = vmap(_syndrome_logicals)
        self.all_syndrome, self.all_logicals = batch_syndrome_logicals(
            self.all_errors)
        self.syndrome_idx = jnp.dot(self.all_syndrome, 2**jnp.arange(0, 8))
        self.logicals_idx = jnp.dot(self.all_logicals, 2**jnp.arange(0, 2))

        self.deformation_transformations = jnp.array([
            [0, 1, 2, 3],  # I
            [0, 2, 1, 3],  # X-Y
            [0, 1, 3, 2],  # Y-Z
            [0, 3, 2, 1],  # X-Z
            [0, 2, 3, 1],  # X-Y-Z
            [0, 3, 1, 2],  # X-Z-Y
        ])

    def logical_error_rate(
        self,
        noise_model: jnp.ndarray,
        deformation: jnp.ndarray,
    ):
        # Deform the noise model
        code_deformation = self.deformation_transformations[deformation]
        data_qubit_error_idx = jnp.zeros_like(
            code_deformation, dtype=int) + jnp.arange(0, deformation.shape[0])[:, None]
        noise_model = noise_model[(data_qubit_error_idx, code_deformation)]

        # Calculate the probabilily that we will have a given syndrome with a given logical
        likelyhood = jnp.prod(self._error_to_prob_batch(
            noise_model, self.all_errors), axis=1)
        record = jnp.zeros(shape=(2**8, 4), dtype=jnp.float32)
        record = record.at[(self.syndrome_idx, self.logicals_idx)].add(
            likelyhood)

        # Rearage result acording to most likely decoding
        #    Logical error:            [0,0]     [1,0]     [0,1]     [1,1]
        rearange_lookup = jnp.array(
            [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
        rearange = rearange_lookup[record.argmax(axis=1)]
        record_rearanged = record[(jnp.arange(0, 2**8)[:, None], rearange)]

        # Calculate the logical error rates
        logical_error_rates = record_rearanged.sum(axis=0)
        # Fix order from IXZY to IXYZ
        logical_error_rates = logical_error_rates[jnp.array([0, 1, 3, 2])]
        return logical_error_rates

    def decode(
        self,
        syndrome: jnp.ndarray,
        noise_model: jnp.ndarray,
        deformation: jnp.ndarray,
    ):
        # Deform the noise model
        code_deformation = self.deformation_transformations[deformation]
        data_qubit_error_idx = jnp.zeros_like(
            code_deformation, dtype=int) + jnp.arange(0, deformation.shape[0])[:, None]
        noise_model = noise_model[(data_qubit_error_idx, code_deformation)]

        # Find all the errors with the given syndrome
        active_syndrome_idx = jnp.dot(syndrome, 2**jnp.arange(0, 8))
        possible_errors = self.all_errors[jnp.where(
            self.syndrome_idx == active_syndrome_idx)]

        # Calculate their likelyhood
        likelyhood = jnp.prod(self._error_to_prob_batch(
            noise_model, possible_errors), axis=1)

        # Pick the most likely one as the recovery operation
        recodery = possible_errors[likelyhood.argmax()]
        return recodery

    def decode_batch(
        self,
        syndromes: jnp.ndarray,
        noise_model: jnp.ndarray,
        deformation: jnp.ndarray,
    ):
        batch_decode = vmap(
            self.decode,
            in_axes=[0, None, None],
            out_axes=0
        )
        return batch_decode(syndromes, noise_model, deformation)

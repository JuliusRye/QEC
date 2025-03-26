from quantum_error_correction_code import SurfaceCode
from neural_network import CNNDual
from perfect_maximum_likelihood_decoder import PMLD
from functools import partial
from jax import random, jit, vmap
import jax.numpy as jnp
from abc import ABC, abstractmethod


class EnvironmentBase(ABC):

    def __init__(
        self,
        noise_model: jnp.ndarray,
        code: SurfaceCode,
    ):
        self.num_qubits = code.hx_original.shape[1]
        self.noise_model = noise_model
        self.code = code

    def reset(
        self,
        key,
    ):
        """
        Reset the system to the non-deformed quantum error correction code.
        """
        state = jnp.zeros(shape=(self.num_qubits), dtype=jnp.int32)
        score, key = self._state_score(key, state)
        return state, score, key

    def _state_score(
        self,
        key,
        state: jnp.ndarray
    ):
        """
        Gives the state a score based on it's logical error rate
        """
        error_rate, key = self._get_state_error_rate(key, state)
        return -jnp.log(.99) / error_rate, key

    @abstractmethod
    def _get_state_error_rate(
        self,
        key,
        state: jnp.ndarray,
    ):
        """
        Calculates the score of the state

        returns: score, key
        """
        pass

    def reward(
        self,
        key,
        current_score: float,
        new_state: jnp.ndarray,
    ):
        new_score, key = self._state_score(key, new_state)
        # Let the reward be the relative improvement in logical error rate
        reward = new_score - current_score
        return reward, new_score, key


class EnvironmentCNN(EnvironmentBase):

    def __init__(
        self,
        model: CNNDual,
        model_params,
        noise_model: jnp.ndarray,
        code: SurfaceCode,
        shots: int
    ):
        """
        The environment the RL-agent will interact with. 
        This environment uses a CNN to estimate the logical error rate of a given deformation. 
        This logical error rate is in turn used to score the state 
        and calculate the reward the RL-agent will recieve.

        model: The CNN that will be used to decode errors on a given deformation

        model_params: The finetuned parameters used by the model after training

        noise_model: The Pauli error probabilities used to generate errors for the decoder

        code: The quantum error correction code that is being deformed

        shots: The number of samples given to the CNN for the estimation of the logical error rate
        """
        super().__init__(noise_model, code)
        self.decoder = model
        self.params = model_params
        self.shots = shots

    @partial(jit, static_argnames=("self"))
    def _get_state_error_rate(
        self,
        key,
        state: jnp.ndarray,
    ):
        """
        Calculates the score of the state corresponding to number of qec cycles that can be performed before going below 99% logical fidelity.

        state: Current Clifford deformation
        """
        keys = random.split(key, num=self.shots+1)

        # Generate syndrome data for the deformation
        parity_info = self.code.deformation_parity_info(state)
        errors = vmap(
            self.code.error,
            in_axes=(0, None),
            out_axes=0
        )(keys[:-1], self.noise_model)
        syndrome_img, logicals = vmap(
            self.code.syndrome_img,
            in_axes=(0, None),
            out_axes=0
        )(errors, parity_info)
        deformation_image = self.code.deformation_image(state)[None,:,:,:]

        # Predict the logical error
        predictions = self.decoder.apply_batch(
            self.params,
            syndrome_img[:, None, :, :],
            deformation_image,
        )
        predicted_logicals = (predictions > 0)

        # Compare prediction with the actual logical error
        error_rate = jnp.any(
            logicals != predicted_logicals,
            axis=1
        ).mean()

        return error_rate, keys[-1]


class EnvironmentPML(EnvironmentBase):

    def __init__(
        self,
        noise_model: jnp.ndarray,
        code: SurfaceCode
    ):
        """
        The environment the RL-agent will interact with. 
        This environment uses the PML-decoder to give the logical error rate of a given deformation. 
        This logical error rate is in turn used to score the state 
        and calculate the reward the RL-agent will recieve.

        noise_model: The Pauli error probabilities used to generate errors for the decoder

        code: The quantum error correction code that is being deformed
        """
        super().__init__(noise_model, code)

    @partial(jit, static_argnames=("self"))
    def _get_state_error_rate(
        self,
        key,  # This argument is only for consistency between environments classes
        state: jnp.ndarray,
    ):
        error_rate = PMLD(
            self.code,
            self.noise_model,
            self.code.deformation_parity_info(state)
        ).exact_logical_error_rate()

        return error_rate, key


class EnvironmentNoiselessCNN(EnvironmentBase):

    def __init__(
        self,
        model: CNNDual,
        model_params,
        noise_model: jnp.ndarray,
        code: SurfaceCode,
    ):
        """
        NOTE: This is very memory intensive and should only be used on small codes. Scales like O(4^n) with n the number of data qubits!

        The environment the RL-agent will interact with. 
        This environment uses a CNN to estimate the logical error rate of a given deformation. 
        This logical error rate is in turn used to score the state 
        and calculate the reward the RL-agent will recieve.

        model: The CNN that will be used to decode errors on a given deformation

        model_params: The finetuned parameters used by the model after training

        noise_model: The Pauli error probabilities used to generate errors for the decoder

        code: The quantum error correction code that is being deformed
        """
        super().__init__(noise_model, code)
        self.decoder = model
        self.params = model_params

        n = code.num_data_qubits
        all_error_idxs = jnp.zeros(
            shape=(2**(2*n), 2*n),
            dtype=jnp.int32
        ) + jnp.arange(0, 2**(2*n))[:, None]
        self.all_errors = (all_error_idxs // 2**jnp.arange(0, 2*n)) % 2
        probs = jnp.array([
            1-noise_model.sum(),  # I
            noise_model[0],  # Z
            noise_model[2],  # Y
            noise_model[1],  # Z
        ])
        self.likelihood = jnp.prod(
            probs[self.all_errors[:, :n] + 2*self.all_errors[:, n:]],
            axis=1
        )

    @partial(jit, static_argnames=("self"))
    def _get_state_error_rate(
        self,
        key,  # This argument is only for consistency between environments classes
        state: jnp.ndarray,
    ):
        """
        Calculates the score of the state corresponding to number of qec cycles that can be performed before going below 99% logical fidelity.

        state: Current Clifford deformation
        """

        # Generate syndrome data for the deformation
        parity_info = self.code.deformation_parity_info(state)
        syndrome_img, logicals = vmap(
            self.code.syndrome_img,
            in_axes=(0, None),
            out_axes=0
        )(self.all_errors.reshape(-1, 2, self.code.num_data_qubits), parity_info)
        deformation_image = self.code.deformation_image(state)[None,:,:,:]

        # Predict the logical error
        predictions = self.decoder.apply_batch(
            self.params,
            syndrome_img[:, None, :, :],
            deformation_image,
        )
        predicted_logicals = (predictions > 0)

        # Compare prediction with the actual logical error
        incorrect_prediction = jnp.any(
            logicals != predicted_logicals,
            axis=1
        )
        error_rate = jnp.sum(incorrect_prediction * self.likelihood)

        return error_rate, key


class EnvironmentLookup(EnvironmentBase):

    def __init__(
        self,
        lookup_table: jnp.ndarray,
        num_of_deformations: int,
        code: SurfaceCode,
        equivalence_mapper,
    ):
        """
        NOTE: This is very memory intensive and should only be used on small codes. Scales like O(4^n) with n the number of data qubits!

        The environment the RL-agent will interact with. 
        This environment uses a CNN to estimate the logical error rate of a given deformation. 
        This logical error rate is in turn used to score the state 
        and calculate the reward the RL-agent will recieve.

        lookup_table: Should be an array of size num_of_deformations**n with n being the number of data qubits.
        """
        super().__init__(None, code)
        self.lookup_table = lookup_table
        self.num_of_deformations = num_of_deformations
        self.equivalence_mapper = equivalence_mapper
        assert lookup_table.shape[0] == num_of_deformations**code.num_data_qubits

    @partial(jit, static_argnames=("self"))
    def _get_state_error_rate(
        self,
        key,  # This argument is only for consistency between environments classes
        state: jnp.ndarray,
    ):
        """
        Calculates the score of the state corresponding to number of qec cycles that can be performed before going below 99% logical fidelity.

        state: Current Clifford deformation
        """

        simplified_deformation = self.equivalence_mapper(state)
        deformation_index = jnp.dot(
            simplified_deformation, 
            self.num_of_deformations**jnp.arange(self.code.num_data_qubits)
        )
        error_rate = self.lookup_table[deformation_index]

        return error_rate, key

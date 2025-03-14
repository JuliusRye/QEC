# Allows us to import files from the parent folder
from quantum_error_correction_code import SurfaceCode
from neural_network import CNNDual
from functools import partial
from jax import random, jit, vmap
import jax.numpy as jnp


class Environment():

    def __init__(
        self,
        model: CNNDual,
        model_params,
        noise_model: jnp.ndarray,
        code: SurfaceCode,
        shots: int,
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
        self.num_qubits = code.hx_original.shape[1]
        self.decoder = model
        self.params = model_params
        self.noise_model = noise_model
        self.code = code
        self.shots = shots
        self.L = int(jnp.sqrt(self.num_qubits))

    def reset(
        self,
        key,
    ):
        """
        Reset the system to the non-deformed quantum error correction code.
        """
        state = jnp.zeros(shape=(self.num_qubits), dtype=jnp.int32)
        score, key = self._score_state(key, state)
        return state, score, key

    @partial(jit, static_argnames=("self"))
    def _score_state(
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
        deformation_image = self.code.deformation_image(state)

        predictions = self.decoder.apply_batch(
            self.params,
            syndrome_img[:, None, :, :],
            deformation_image,
        )

        error_rate = jnp.any(
            logicals != (predictions > 0),
            axis=1
        ).mean()

        # Number of qec cycles that can be performed while maintaining above 99% logical fidelity
        score = -jnp.log(.99) / error_rate

        return score, keys[-1]

    def reward(
        self,
        key,
        current_score: float,
        new_state: jnp.ndarray,
    ):
        new_score, key = self._score_state(key, new_state)
        reward = new_score - current_score
        return reward, new_score, key

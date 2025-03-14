# Allows us to import files from the parent folder
from quantum_error_correction_code import QEC
from perfect_maximum_likelihood_decoder import PMLD
from functools import partial
from jax import jit
import jax.numpy as jnp

class Environment():

    def __init__(
        self,
        noise_model: jnp.ndarray,
        code: QEC,
    ):
        """
        The environment the RL-agent will interact with. 
        This environment uses the PML-decoder to give the logical error rate of a given deformation. 
        This logical error rate is in turn used to score the state 
        and calculate the reward the RL-agent will recieve.

        noise_model: The Pauli error probabilities used to generate errors for the decoder

        code: The quantum error correction code that is being deformed
        """
        self.num_qubits = code.hx_original.shape[1]
        self.noise_model = noise_model
        self.code = code
        self.L = int(jnp.sqrt(self.num_qubits))
        self.scores = jnp.load("data/environmentPML_scores_p01_nu500_halve.npy")

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
        key,  # This argument is only for consistency between environments classes
        state: jnp.ndarray,
    ):
        """
        Calculates the score of the state corresponding to number of qec cycles that can be performed before going below 99% logical fidelity.

        state: Current Clifford deformation
        """

        # error_rate = PMLD(
        #     self.code,
        #     self.noise_model,
        #     self.code.deformation_parity_info(state)
        # ).exact_logical_error_rate()

        # # Number of qec cycles that can be performed while maintaining above 99% logical fidelity
        # score = -jnp.log(.99) / error_rate

        idx = jnp.dot(jnp.array([0,0,1,2,2,1])[state], 3**jnp.arange(9))
        score = self.scores[idx]

        return score, key

    def reward(
        self,
        key,
        current_score: float,
        new_state: jnp.ndarray,
    ):
        new_score, key = self._score_state(key, new_state)
        reward = new_score - current_score
        return reward, new_score, key

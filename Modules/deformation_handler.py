import jax.numpy as jnp
from abc import ABC, abstractmethod
from jax import random, jit, vmap
from functools import partial
from neural_network import CNNDual
from quantum_error_correction_code import QEC


class DeformationManager:

    def __init__(
        self,
        deformation_options: jnp.ndarray,
        L: int,
    ):
        """
        Used creating random deformations and turning a deformation into an image for a convolutional neural network.

        deformation_options: Array of the index's of the allowed deformations

        L: Code distance of the surface code
        """
        self.deformation_options = deformation_options
        self.L = L

    @partial(jit, static_argnames=("self"))
    def random_deformation(
        self,
        key,
    ):
        """
        Creates a random deformation.

        returns: tuple of (deformation, key)
        """
        subkey, key = random.split(key)
        deformation = self.deformation_options[random.randint(
            subkey,
            shape=(self.L**2),
            minval=0,
            maxval=self.deformation_options.shape[0]
        )]
        return deformation, key

    @partial(jit, static_argnames=("self"))
    def deformation_image(
        self,
        deformation: jnp.ndarray,
    ):
        """
        Converts a surface code deformation into an image that can be given to the CNN Dual neural network.

        deformation: int array of shape (code_distance**2)

        returns: float Matrix of shape (1, 6, code_distance, code_distance). // Batch size | Image channels | width | height

        Channel n corresponds to clifford deformation n, with ones on the data qubits that use that deformation and zero on the rest.
        """
        img_deformation = jnp.eye(6, dtype=jnp.float32)[
            deformation.reshape((self.L, self.L))
        ].transpose(2, 0, 1)
        # return img_deformation[None,:,:,:]
        img_deformation_roll = jnp.roll(img_deformation, shift=3, axis=0)
        mask = (jnp.arange(self.L)[:, None] +
                jnp.arange(self.L)[None, :]) % 2 == 1
        return jnp.where(mask[None, :, :], img_deformation, img_deformation_roll)[None, :, :, :]

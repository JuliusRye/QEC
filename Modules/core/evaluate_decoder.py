from quantum_error_correction_code import SurfaceCode
from neural_network import CNNDual, CNNDecoder
from perfect_maximum_likelihood_decoder import PMLD
from pymatching import Matching
from jax import random, vmap, jit
import jax.numpy as jnp

# Helper functions for evaluating the decoder

def data_batch(
    key,
    code: SurfaceCode,
    batch_size: int,
    parity_info: tuple[jnp.ndarray],
    error_probs: jnp.ndarray,
    as_images: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, any]:
    keys = random.split(key, num=batch_size)
    errors = vmap(
        code.error,
        in_axes=(0, None),
        out_axes=0
    )(keys, error_probs)
    if as_images:
        syndromes, logicals = vmap(
            code.syndrome_img,
            in_axes=(0, None),
            out_axes=0
        )(errors, parity_info)
    else:
        syndromes, logicals = vmap(
            code.syndrome,
            in_axes=(0, None),
            out_axes=0
        )(errors, parity_info)
    return syndromes, logicals

def evaluate_cnn_decoder(
    key,
    decoder_model: CNNDual | CNNDecoder,
    model_params: dict,
    code: SurfaceCode,
    deformation: jnp.ndarray,
    batch_count: int,
    batch_size: int,
    error_probs: jnp.ndarray,
):
    """
    Evaluate the CNN decoder on batch_count * batch_size samples.

    NOTE: deformation image is only needed for the `decoder_model` of type `CNNDual`
    """
    @jit
    def evaluate_batch(
        key,
        parity_info: tuple[jnp.ndarray]
    ):
        syndromes, logicals = data_batch(
            key,
            code,
            batch_size,
            parity_info,
            error_probs,
            as_images=True
        )
        result = decoder_model.apply_batch(model_params, syndromes[:,None,:,:])
        predictions = result > 0.0
        logical_error_rate = (predictions != logicals).any(axis=1).mean()
        i = 2 * logicals[:, 0] + logicals[:, 1]
        j = 2 * predictions[:, 0] + predictions[:, 1]
        return logical_error_rate, i, j
    
    parity_info = code.deformation_parity_info(deformation)
    keys = random.split(key, num=batch_count)
    logical_error_rate = jnp.empty((batch_count,), dtype=jnp.float32)
    correlation_hist2d = jnp.zeros((4, 4), dtype=jnp.int32)
    for r, subkey in enumerate(keys):
        ler, i, j = evaluate_batch(subkey, parity_info)
        logical_error_rate = logical_error_rate.at[r].set(ler)
        correlation_hist2d = correlation_hist2d.at[j, i].add(1)
    return logical_error_rate.mean(), correlation_hist2d

def evaluate_mcnn_decoder(
    key,
    decoder_model: CNNDual | CNNDecoder,
    model_params: dict,
    code: SurfaceCode,
    deformation: jnp.ndarray,
    batch_count: int,
    batch_size: int,
    error_probs: jnp.ndarray,
):
    """
    Evaluate the CNN decoder on batch_count * batch_size samples.

    NOTE: deformation image is only needed for the `decoder_model` of type `CNNDual`
    """
    @jit
    def evaluate_batch(
        key,
        parity_info: tuple[jnp.ndarray],
        deformation_image: jnp.ndarray,
    ):
        syndromes, logicals = data_batch(
            key,
            code,
            batch_size,
            parity_info,
            error_probs,
            as_images=True
        )
        result = decoder_model.apply_batch(model_params, syndromes[:,None,:,:], deformation_image[None,:,:,:])
        predictions = result > 0.0
        logical_error_rate = (predictions != logicals).any(axis=1).mean()
        i = 2 * logicals[:, 0] + logicals[:, 1]
        j = 2 * predictions[:, 0] + predictions[:, 1]
        return logical_error_rate, i, j
    
    parity_info = code.deformation_parity_info(deformation)
    deformation_image = code.deformation_image(deformation)
    keys = random.split(key, num=batch_count)
    logical_error_rate = jnp.empty((batch_count,), dtype=jnp.float32)
    correlation_hist2d = jnp.zeros((4, 4), dtype=jnp.int32)
    for r, subkey in enumerate(keys):
        ler, i, j = evaluate_batch(subkey, parity_info, deformation_image)
        logical_error_rate = logical_error_rate.at[r].set(ler)
        correlation_hist2d = correlation_hist2d.at[j, i].add(1)
    return logical_error_rate.mean(), correlation_hist2d

def evaluate_pml_decoder(
    key,
    code: SurfaceCode,
    deformation: jnp.ndarray,
    batch_count: int,
    batch_size: int,
    error_probs: jnp.ndarray,
    exact: bool = False,
):
    """
    Evaluate the PML decoder on batch_count * batch_size samples.

    If exact is True, the exact numbers are returned and batch_count and batch_size are ignored.    
    """
    def evaluate_batch(
        key,
        decoder: PMLD,
        parity_info: tuple[jnp.ndarray]
    ):
        syndromes, logicals = data_batch(
            key,
            code,
            batch_size,
            parity_info,
            error_probs,
            as_images=False
        )
        predictions = decoder.decode_batch(syndromes)
        logical_error_rate = (predictions != logicals).any(axis=1).mean()
        i = 2 * logicals[:, 0] + logicals[:, 1]
        j = 2 * predictions[:, 0] + predictions[:, 1]
        return logical_error_rate, i, j
    
    parity_info = code.deformation_parity_info(deformation)
    decoder = PMLD(code, error_probs, parity_info)

    if exact:
        return decoder.exact_logical_error_rate(), decoder.hist2d

    keys = random.split(key, num=batch_count)
    logical_error_rate = jnp.empty((batch_count,), dtype=jnp.float32)
    correlation_hist2d = jnp.zeros((4, 4), dtype=jnp.int32)
    for r, subkey in enumerate(keys):
        ler, i, j = evaluate_batch(subkey, decoder, parity_info)
        logical_error_rate = logical_error_rate.at[r].set(ler)
        correlation_hist2d = correlation_hist2d.at[j, i].add(1)
    return logical_error_rate.mean(), correlation_hist2d

def logicals_of_recovery(
    code: SurfaceCode,
    recovery: jnp.ndarray,
    parity_info: tuple[jnp.ndarray],
) -> jnp.ndarray:
    _, logicals = vmap(
        code.syndrome,
        in_axes=(0, None),
        out_axes=0
    )(recovery, parity_info)
    return logicals

def evaluate_mwpm_decoder(
    key,
    code: SurfaceCode,
    deformation: jnp.ndarray,
    batch_count: int,
    batch_size: int,
    error_probs: jnp.ndarray,
):
    """
    Evaluate the MWPM decoder on batch_count * batch_size samples.
    """
    def evaluate_batch(
        key,
        decoder: Matching,
        parity_info: tuple[jnp.ndarray],
        parity_info_CSS: tuple[jnp.ndarray]
    ):
        syndromes, logicals = data_batch(
            key,
            code,
            batch_size,
            parity_info,
            error_probs,
            as_images=False
        )
        recovery = decoder.decode_batch(syndromes)
        # Reshape to seperate x and z indices
        recovery = recovery.reshape((recovery.shape[0], 2, recovery.shape[1]//2))
        _, predictions = vmap(
            code.syndrome,
            in_axes=(0, None),
            out_axes=0
        )(recovery, parity_info_CSS)
        logical_error_rate = (predictions != logicals).any(axis=1).mean()
        i = 2 * logicals[:, 0] + logicals[:, 1]
        j = 2 * predictions[:, 0] + predictions[:, 1]
        return logical_error_rate, i, j
    
    parity_info = code.deformation_parity_info(deformation)
    parity_info_CSS = code.deformation_parity_info(jnp.zeros(code.num_data_qubits, dtype=jnp.int32))
    decoder = Matching(
        jnp.append(parity_info_CSS[0], parity_info_CSS[1], axis=1)
    )
    keys = random.split(key, num=batch_count)
    logical_error_rate = jnp.empty((batch_count,), dtype=jnp.float32)
    correlation_hist2d = jnp.zeros((4, 4), dtype=jnp.int32)
    for r, subkey in enumerate(keys):
        ler, i, j = evaluate_batch(subkey, decoder, parity_info, parity_info_CSS)
        logical_error_rate = logical_error_rate.at[r].set(ler)
        correlation_hist2d = correlation_hist2d.at[j, i].add(1)
    return logical_error_rate.mean(), correlation_hist2d

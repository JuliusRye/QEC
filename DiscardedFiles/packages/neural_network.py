import jax.numpy as jnp
import jax

from icecream import ic  # For debugging

import json


def save_NN(file: str, params: dict[str, list[jnp.ndarray]]) -> None:
    package = {key: [val.tolist() for val in vals]
               for key, vals in params.items()}
    with open(file, 'w') as file:
        json.dump(package, file, indent=4)


def load_NN(file: str) -> dict[str, list[jnp.ndarray]]:
    with open(file, 'r') as file:
        package = json.load(file)
    return {key: [jnp.array(val) for val in vals] for key, vals in package.items()}


def sigmoid(x):
    return 1/(1 + jnp.exp(-x))


def soft_plus(x):
    return jnp.log2(1+jnp.exp(x))


def NN_init_params(key, num_neurons_layers):
    """
    Given a jax random key and a list of the neuron numbers
    in the layers of a network (simple fully connected network,
    i.e. 'multi-layer perceptron'), return a dictionary
    with the weights initialized randomly and biases set to zero.

    Returns: params, with params['weights'] a list of matrices and
    params['biases'] a list of vectors.
    """
    params = {}
    params['weights'] = []
    params['biases'] = []

    for lower_layer, higher_layer in zip(num_neurons_layers[:-1], num_neurons_layers[1:]):
        key, subkey = jax.random.split(key)
        params['weights'].append(jax.random.normal(subkey,
                                                   [higher_layer, lower_layer]) /
                                 jnp.sqrt(lower_layer))

    for num_neurons in num_neurons_layers[1:]:
        params['biases'].append(jnp.zeros(num_neurons))

    return params


def NN(x, params):
    """
    Standard multilayer perception "MLP" with params['weights'] and params['biases'],
    applied to input vector x. Activation tanh applied to all
    layers except last.

    Returns activation vector of the output layer.
    """

    num_layers = len(params['weights'])
    for layer_idx, (w, b) in enumerate(zip(params['weights'], params['biases'])):
        x = jnp.matmul(w, x) + b
        if layer_idx < num_layers-1:
            x = jnp.tanh(x)
        else:
            x = sigmoid(x)
            # x = soft_plus(x)
    return x


NN_batch = jax.vmap(NN, in_axes=[0, None], out_axes=0)


def NN_raw_to_correction(NN_output: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize the NN output to represent probabilities 
    using the exponentiel of the values as their respective weights
    """
    # logits have values between zero and one
    logits = NN_output.reshape((4, NN_output.shape[0]//4))
    # Normalize so probabilities sum to one
    probabilities = logits / jnp.sum(logits, axis=0)
    return probabilities


NN_raw_to_correction_batch = jax.vmap(
    NN_raw_to_correction,
    in_axes=0,
    out_axes=0
)


def NN_correction_to_syndrome(NN_correction: jnp.ndarray, Mx: jnp.ndarray, My: jnp.ndarray, Mz: jnp.ndarray) -> jnp.ndarray:
    """
    Uses the probabilities of the NN_correction to calculate
    the probability that each of the stabilizers will be activated.

    This function does not take into account the correlation between
    stabilizers and merely evaluates each stabilizer independently
    """
    # Note: this is not matrix multiplication but element wise multiplication
    probabilities_x = Mx * NN_correction[1]
    probabilities_y = My * NN_correction[2]
    probabilities_z = Mz * NN_correction[3]
    probabilities = probabilities_x + probabilities_y + probabilities_z
    # Calculate probability that the stabilizer is activated
    p1 = jnp.zeros(probabilities.shape[0])
    for p in probabilities.T:
        #      0 -> 1    1 -> 1
        p1 = (1-p1)*p + p1*(1-p)
    return p1


NN_correction_to_syndrome_batch = jax.vmap(
    NN_correction_to_syndrome,
    in_axes=(0, None, None, None),
    out_axes=0
)


def mse_loss_batch(syndromes: jnp.ndarray, params: dict, Mx: jnp.ndarray, My: jnp.ndarray, Mz: jnp.ndarray) -> jnp.ndarray:
    """
    Uses the mean square to evaluate the performance of the NN
    """
    # Calculate the syndome probabilities produced by the correction given by the NN
    NN_raws = NN_batch(syndromes[:, :-2], params)
    NN_corrections = NN_raw_to_correction_batch(NN_raws)
    NN_syndromes = NN_correction_to_syndrome_batch(NN_corrections, Mx, My, Mz)
    # Return the mean square of the NN_syndromes vs the measured syndromes
    # Add a small number so the loss never reaches zero as that seams to break the optimizer
    return jnp.mean((NN_syndromes - syndromes)**2) + 1E-20


mse_loss_batch_val_grad = jax.value_and_grad(mse_loss_batch, argnums=1)
mse_loss_batch_val_grad = jax.jit(mse_loss_batch_val_grad)

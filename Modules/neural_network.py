import jax.numpy as jnp
from jax import random, nn, jit, lax, vmap
from functools import partial
from abc import ABC, abstractmethod
import json


def save_params(
    params: dict | list | jnp.ndarray,
    file_name: str,
):
    """
    Saves the neural network parameter object to a json file.
    """
    def jsonify_params(params: dict | list | jnp.ndarray):
        if isinstance(params, dict):
            return {k: jsonify_params(v) for k, v in params.items()}
        if isinstance(params, list):
            return [jsonify_params(v) for v in params]
        if isinstance(params, jnp.ndarray):
            return params.tolist()
        raise NotImplementedError(
            f"Handling of type {type(params)} has not been implemented")
    with open(file_name, 'w') as file:
        json.dump(jsonify_params(params), file, indent=4)


def load_params(
    file_name: str,
) -> dict | list | jnp.ndarray:
    """
    Loads the neural network parameter object from a JSON file.
    """
    def dejsonify_params(params: dict | list | jnp.ndarray):
        if isinstance(params, dict):
            return {k: dejsonify_params(v) for k, v in params.items()}
        if isinstance(params, list):
            try:
                return jnp.array(params)
            except TypeError:
                return [dejsonify_params(v) for v in params]
        raise NotImplementedError(
            f"Handling of type {type(params)} has not been implemented")
    with open(file_name, 'r') as file:
        return dejsonify_params(json.load(file))


class MLModel(ABC):

    @abstractmethod
    def init(
        self,
        key
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """
        This initializes all the parameters in the neural network to random values.

        key: Used to generate the random parameters in the neural network in a deterministic manner
        """
        pass

    @abstractmethod
    def apply_batch(
        self,
        params: list[tuple[jnp.ndarray, jnp.ndarray]],
        x: jnp.ndarray
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def apply_single(
        self,
        params: list[tuple[jnp.ndarray, jnp.ndarray]],
        x: jnp.ndarray
    ) -> jnp.ndarray:
        pass


class MLP(MLModel):
    def __init__(
        self,
        layer_sizes: list[int],
        activation_on_last_layer=False,
    ):
        """
        layer_sizes: List of ints

        activation_on_last_layer: Bool for whether or not to use the activation function on all layers or to exclude the last layer
        """
        self.layer_sizes = layer_sizes
        self.activation_on_last_layer = activation_on_last_layer

    def init(
        self,
        key,
    ) -> list[dict[str, jnp.ndarray]]:
        """Initializes the MLP parameters with random weights and biases."""
        params = []
        keys = random.split(key, len(self.layer_sizes) - 1)
        for i, k in enumerate(keys):
            w = random.normal(
                k, (self.layer_sizes[i], self.layer_sizes[i + 1])) * jnp.sqrt(2.0 / self.layer_sizes[i])
            b = jnp.zeros((self.layer_sizes[i + 1],))
            params.append({'w': w, 'b': b})
        return params

    @partial(jit, static_argnames=("self"))
    def _apply_batch(
        self,
        params: list[dict[str, jnp.ndarray]],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        def _apply_single(x):
            for layer in params[:-1]:
                x = nn.relu(jnp.dot(x, layer['w']) + layer['b'])
            layer = params[-1]
            x = jnp.dot(x, layer['w']) + layer['b']
            return nn.relu(x) if self.activation_on_last_layer else x
        return vmap(_apply_single)(x)

    def apply_single(
        self,
        params: list[dict[str, jnp.ndarray]],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        #### Jit optimized function!

        Applies the MLP to a single input x.

        x: array of shape (input_size)

        returns: array of shape (output_size)
        """
        return self._apply_batch(params, x[None, :])[0]

    def apply_batch(
        self,
        params: list[dict[str, jnp.ndarray]],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        #### Jit optimized function!

        Applies the MLP to a batch of inputs x.

        x: array of shape (batch_size, input_size)

        returns: array of shape (batch_size, output_size)
        """
        return self._apply_batch(params, x)


class CNN(MLModel):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        conv_layers: list[tuple[int, int, int, int]],
        activation_on_last_layer=False,
    ):
        """
        input_shape: Tuple of ints (input_channels, input_height, input_width)

        conv_layers: List of tuples (num_filters, kernel_size, stride, pad)

        activation_on_last_layer: Bool for whether or not to use the activation function on all layers or to exclude the last layer
        """
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.activation_on_last_layer = activation_on_last_layer
        # Calculate the layer sizes
        self.layer_sizes = [self.input_shape]
        layer_height, layer_width = self.input_shape[1:]
        for num_filters, kernel_size, stride, pad in self.conv_layers:
            layer_height = (layer_height - kernel_size + 2*pad) // stride + 1
            layer_width = (layer_width - kernel_size + 2*pad) // stride + 1
            self.layer_sizes.append((num_filters, layer_height, layer_width))
        if jnp.array(self.layer_sizes).min() <= 0:
            raise ValueError(
                f"Some layer dimentions are zero or negative. Layer sizes = {self.layer_sizes}")

    def init(
        self,
        key,
    ) -> list[dict[str, jnp.ndarray]]:
        """Initializes the CNN parameters with random weights and biases."""
        params = []
        keys = random.split(
            key,
            len(self.conv_layers)
        )

        # Initialize convolutional layers
        in_channels = self.input_shape[0]
        for i, (num_filters, kernel_size, stride, pad) in enumerate(self.conv_layers):
            k = keys[i]
            w = random.normal(k, (num_filters, in_channels, kernel_size, kernel_size)) * \
                jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
            b = jnp.zeros((num_filters,))
            params.append({'w': w, 'b': b})
            in_channels = num_filters

        return params

    @partial(jit, static_argnames=("self"))
    def _apply_batch(
        self,
        params: list[dict[str, jnp.ndarray]],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        for layer, (_, _, stride, pad) in zip(params[:-1], self.conv_layers[:-1]):
            x = jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
            x = lax.conv_general_dilated(
                lhs=x,
                rhs=layer['w'],
                window_strides=(stride, stride),
                padding='VALID',
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            ) + layer['b'].reshape(-1, 1, 1)
            x = nn.relu(x)

        layer = params[-1]
        _, _, stride, pad = self.conv_layers[-1]
        x = jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=layer['w'],
            window_strides=(stride, stride),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        ) + layer['b'].reshape(-1, 1, 1)
        return nn.relu(x) if self.activation_on_last_layer else x

    def apply_single(
        self,
        params: list[dict[str, jnp.ndarray]],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        #### Jit optimized function!

        Applies the CNN to a single input x.

        x: array of shape (input_channels, input_width, input_height)

        returns: array of shape (output_channels, output_width, output_height)
        """
        return self._apply_batch(params, x[None, :])[0]

    def apply_batch(
        self,
        params: list[dict[str, jnp.ndarray]],
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        #### Jit optimized function!

        Applies the CNN to a batch of inputs x.

        x: array of shape (batch_size, input_channels, input_width, input_height)

        returns: array of shape (batch_size, output_channels, output_width, output_height)
        """
        return self._apply_batch(params, x)


class CNNDual(MLModel):
    def __init__(
        self,
        input_shape_1: tuple[int, int, int],
        input_shape_2: tuple[int, int, int],
        conv_layers_input_1: list[tuple[int, int, int, int]],
        conv_layers_input_2: list[tuple[int, int, int, int]],
        conv_layers_stage_2: list[tuple[int, int, int, int]],
        fc_layers: list[int],
    ):
        """
        Version of the CNN class that takes two images of potentially different dimentions as input instead of one.

        NOTE: Stride should be 1 for the first convolutional layer in order for the dimentions to work out.
        For the same reason the size difference in the two dimentions between the images should also be the same.

        input_shape_1: Tuple of ints (input_channels, input_height, input_width)

        input_shape_2: Tuple of ints (input_channels, input_height, input_width)

        conv_layers_input_1: List of tuples (num_filters, kernel_size, stride, pad) 

        conv_layers_input_2: List of tuples (num_filters, kernel_size, stride, pad) 

        conv_layers_stage_2: List of tuples (num_filters, kernel_size, stride, pad) 

        fc_layers: List of fully connected layer sizes
        """
        # Create the sub models
        self.sub_models: dict[str, MLModel] = {}
        # Stage 1
        self.sub_models['CNN_input_1'] = CNN(
            input_shape_1,
            conv_layers_input_1,
            activation_on_last_layer=False
        )
        self.sub_models['CNN_input_2'] = CNN(
            input_shape_2,
            conv_layers_input_2,
            activation_on_last_layer=False
        )
        # Stage 2
        self.sub_models['CNN_stage_2'] = CNN(
            self.sub_models['CNN_input_1'].layer_sizes[-1],
            conv_layers_stage_2,
            activation_on_last_layer=True
        )
        # Stage 3
        cnn_prim_output_neurons = int(jnp.prod(jnp.array(
            self.sub_models['CNN_stage_2'].layer_sizes[-1]
        )))
        self.sub_models['MLP_stage_3'] = MLP(
            [cnn_prim_output_neurons, *fc_layers],
            activation_on_last_layer=False
        )
        self.layer_sizes = {
            'CNN_input_1': self.sub_models['CNN_input_1'].layer_sizes,
            'CNN_input_2': self.sub_models['CNN_input_2'].layer_sizes,
            'CNN_stage_2': self.sub_models['CNN_stage_2'].layer_sizes,
            'MLP_stage_3': self.sub_models['MLP_stage_3'].layer_sizes,
        }
        # Check for mathcing dimentions of the output in stage 1
        if self.layer_sizes['CNN_input_1'][-1] != self.layer_sizes['CNN_input_2'][-1]:
            for name, layer_sizes in self.layer_sizes.items():
                print(name, layer_sizes)
            raise ValueError(
                f"The outputs of the first two CNN's do not result in matching dimentions. The outputs are {self.layer_sizes['CNN_input_1'][-1]} != {self.layer_sizes['CNN_input_2'][-1]}")

    def init(
        self,
        key,
    ) -> dict[str, list[dict[str, jnp.ndarray]]]:
        """Initializes the MLP parameters with random weights and biases."""
        keys = random.split(key, num=4)
        params = {}
        params['CNN_input_1'] = self.sub_models['CNN_input_1'].init(keys[0])
        params['CNN_input_2'] = self.sub_models['CNN_input_2'].init(keys[1])
        params['CNN_stage_2'] = self.sub_models['CNN_stage_2'].init(keys[2])
        params['MLP_stage_3'] = self.sub_models['MLP_stage_3'].init(keys[3])
        return params

    # @partial(jit, static_argnames=("self"))
    def _apply_batch(
        self,
        params: dict[str, list[dict[str, jnp.ndarray]]],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        # Stage 1
        x1 = self.sub_models['CNN_input_1'].apply_batch(
            params['CNN_input_1'], x1
        )
        x2 = self.sub_models['CNN_input_2'].apply_batch(
            params['CNN_input_2'], x2
        )
        # Merge
        x = x1*x2
        # Stage 2
        x = self.sub_models['CNN_stage_2'].apply_batch(
            params['CNN_stage_2'], x
        )
        # Flatten the image (but not over different batches)
        x = x.reshape(x.shape[0], -1)
        # Stage 3
        x = self.sub_models['MLP_stage_3'].apply_batch(
            params['MLP_stage_3'], x
        )
        return x

    def apply_single(
        self,
        params: dict[str, list[dict[str, jnp.ndarray]]],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        #### Jit optimized function!

        Applies the CNN Dual to a single input x.

        x1: array of shape (input_channels_1, input_width_1, input_height_1)

        x2: array of shape (input_channels_2, input_width_2, input_height_2)

        returns: array of shape (output_size)
        """
        return self._apply_batch(params, x1[None, :, :, :], x2[None, :, :, :])[0]

    def apply_batch(
        self,
        params: dict[str, list[dict[str, jnp.ndarray]]],
        x1: jnp.ndarray,
        x2: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        #### Jit optimized function!

        Applies the CNN Dual to a batch of inputs x.

        x1: array of shape (batch_size, input_channels_1, input_width_1, input_height_1)

        x2: array of shape (batch_size, input_channels_2, input_width_2, input_height_2)

        returns: array of shape (batch_size, output_size)
        """
        return self._apply_batch(params, x1, x2)

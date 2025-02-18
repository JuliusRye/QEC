import jax.numpy as jnp
from jax import random, nn, jit, lax, vmap
from functools import partial


class MLP:
    def __init__(
        self,
        layer_sizes: list[int]
    ):
        self.layer_sizes = layer_sizes

    def init(
        self,
        key
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """Initializes the MLP parameters with random weights and biases."""
        params = []
        keys = random.split(key, len(self.layer_sizes) - 1)
        print(f"Input vector of size {self.layer_sizes[0]}")
        for i, k in enumerate(keys):
            w = random.normal(
                k, (self.layer_sizes[i], self.layer_sizes[i + 1])) * jnp.sqrt(2.0 / self.layer_sizes[i])
            b = jnp.zeros((self.layer_sizes[i + 1],))
            params.append((w, b))
            print(f"to vector of size {self.layer_sizes[i + 1]}")
        return params

    @partial(jit, static_argnames=("self"))
    def apply(
        self,
        params: list[tuple[jnp.ndarray, jnp.ndarray]],
        x: jnp.ndarray
    ) -> jnp.ndarray:
        """Applies the MLP to an input x."""
        @vmap
        def _apply_single(
            x
        ):
            for w, b in params[:-1]:
                x = nn.relu(jnp.dot(x, w) + b)
            w, b = params[-1]
            return jnp.dot(x, w) + b  # No activation in the output layer
        return _apply_single(x)


class CNN:
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        conv_layers: list[tuple[int, int, int]],
        fc_layers: list[int],
    ):
        """
        input_shape: Tuple of ints (input_channels, input_height, input_width)

        conv_layers: List of tuples (num_filters, kernel_size, stride)

        fc_layers: List of fully connected layer sizes
        """
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers

    def init(
        self,
        key,
    ) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
        """Initializes the CNN parameters with random weights and biases."""
        params = []
        keys = random.split(key, len(self.conv_layers) +
                            len(self.fc_layers) - 1)

        # Initialize convolutional layers
        in_channels = self.input_shape[0]
        for i, (num_filters, kernel_size, stride) in enumerate(self.conv_layers):
            k = keys[i]
            w = random.normal(k, (num_filters, in_channels, kernel_size, kernel_size)) * \
                jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
            b = jnp.zeros((num_filters,))
            params.append((w, b))
            in_channels = num_filters

        # Flatten output size estimation (assuming valid padding)
        conv_out_height, conv_out_width = self.input_shape[1], self.input_shape[2]
        print(
            f"Input image of size {conv_out_height} by {conv_out_width} with {self.input_shape[0]} channels")
        for num_filters, kernel_size, stride in self.conv_layers:
            conv_out_height = (conv_out_height - kernel_size) // stride + 1
            conv_out_width = (conv_out_width - kernel_size) // stride + 1
            print(
                f"to image of size {conv_out_height} by {conv_out_width} with {num_filters} channels")

        # Initialize fully connected layers
        fc_input_size = conv_out_height * conv_out_width * in_channels
        layer_sizes = [fc_input_size] + self.fc_layers
        print(f"reshaped to vector of size {fc_input_size}")
        for i in range(len(self.fc_layers)):
            k = keys[len(self.conv_layers) + i]
            w = random.normal(
                k, (layer_sizes[i], layer_sizes[i + 1])) * jnp.sqrt(2.0 / layer_sizes[i])
            b = jnp.zeros((layer_sizes[i + 1],))
            params.append((w, b))
            print(f"to vector of size {layer_sizes[i + 1]}")

        return params

    @partial(jit, static_argnames=("self"))
    def apply(
        self,
        params: list[tuple[jnp.ndarray, jnp.ndarray]],
        x: jnp.ndarray
    ) -> jnp.ndarray:
        """Applies the CNN to an input x."""
        param_idx = 0

        # Apply convolutional layers
        for (w, b), (_, _, stride) in zip(params[:len(self.conv_layers)], self.conv_layers):
            x = lax.conv_general_dilated(
                lhs=x,
                rhs=w,
                window_strides=(stride, stride),
                padding='VALID',
                dimension_numbers=('NCHW', 'OIHW', 'NCHW')
            )
            x = nn.relu(x + b.reshape(-1, 1, 1))
            param_idx += 1

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Apply fully connected layers
        for w, b in params[param_idx:-1]:
            x = nn.relu(jnp.dot(x, w) + b)

        # Output layer (no activation)
        w, b = params[-1]
        return jnp.dot(x, w) + b

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, jit, vmap
from functools import partial
from matplotlib.colors import ListedColormap


class QEC:

    def __init__(
        self,
        data_qubit_loc: jnp.ndarray,
        syndrome_qubit_loc: jnp.ndarray,
        parity_check_matrix: jnp.ndarray,
        logical_parity_matrix: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        """
        data_qubit_loc: shape=(n,2) array of 2d location of n data qubits
        syndrome_qubit_loc: shape=(s,2) array of 2d location of s syndrome qubits
        parity_check_matrix: A matrix of shape=(2,s,n) of parity check X and Z
        mask: A 2d array with true false values to map the syndrome onto (Number of true values should equal number of syndrome qubits)
        """
        self.data_qubit_loc = data_qubit_loc
        self.syndrome_qubit_loc = syndrome_qubit_loc

        self.hx_original, self.hz_original = parity_check_matrix
        self.lx_original, self.lz_original = logical_parity_matrix

        self.deformation = jnp.zeros(
            shape=self.hx_original.shape[1], dtype=jnp.int32)
        self.mask = mask

    def deformation_parity_info(
        self,
        D: jnp.ndarray,
    ) -> tuple[jnp.ndarray]:

        transformations = jnp.array([
            [[1, 0], [0, 1]],  # I
            [[1, 1], [0, 1]],  # X-Y
            [[1, 0], [1, 1]],  # Y-Z
            [[0, 1], [1, 0]],  # X-Z
            [[0, 1], [1, 1]],  # X-Y-Z
            [[1, 1], [1, 0]],  # X-Z-Y
        ])

        A = jnp.append(self.hx_original, self.lx_original, axis=0)
        B = jnp.append(self.hz_original, self.lz_original, axis=0)

        # Apply transformation column-wise
        A_prime, B_prime = vmap(
            lambda A, B, Di: jnp.dot(Di, jnp.stack([A, B])) % 2,
            in_axes=(1, 1, 0),
            out_axes=2
        )(
            A,
            B,
            transformations[D]
        )

        m, n = self.hx_original.shape

        hx = A_prime[:m]
        lx = A_prime[m:]

        hz = B_prime[:m]
        lz = B_prime[m:]

        parity_info = (hx, hz, lx, lz)
        return parity_info

    @partial(jit, static_argnames=("self"))
    def error(
        self,
        key,
        probabilities: jnp.ndarray,
        parity_info: tuple[jnp.ndarray],
    ) -> tuple[jnp.ndarray]:
        hx, hz, lx, lz = parity_info
        px, py, pz = probabilities
        n = hx.shape[1]
        rv = random.uniform(key, shape=n)
        error_x = rv < (px + py)
        error_z = jnp.logical_and(
            rv > px,
            rv < px + py + pz
        )
        return jnp.column_stack((
            error_x,
            error_z
        )).T

    @partial(jit, static_argnames=("self"))
    def syndrome(
        self,
        error: jnp.ndarray,
        parity_info: tuple[jnp.ndarray],
    ) -> tuple[jnp.ndarray]:
        hx, hz, lx, lz = parity_info
        # Calculate syndrome
        parity_x = jnp.matmul(hx, error[0])
        parity_z = jnp.matmul(hz, error[1])
        syndrome = (parity_x + parity_z) % 2
        # Calculate logicals
        parity_x = jnp.matmul(lx, error[0])
        parity_z = jnp.matmul(lz, error[1])
        logicals = (parity_x + parity_z) % 2
        return syndrome, logicals

    @partial(jit, static_argnames=("self"))
    def syndrome_img(
        self,
        error: (jnp.ndarray),
        parity_info: tuple[jnp.ndarray],
    ) -> tuple[jnp.ndarray]:
        """
        Creates an image of the syndrome from the error with:

        -1: Detected a syndrome here

        0: No syndrome measurement at this location

        1: Detected no syndrome
        """
        syndrome, logicals = self.syndrome(error, parity_info)
        syndrome = syndrome.astype(jnp.int32)
        img = jnp.zeros_like(
            self.mask,
            dtype=jnp.float32
        ).at[self.mask].set(1 - 2*syndrome)
        return img, logicals

    def show(
        self,
        parity_info: tuple[jnp.ndarray],
        error: tuple[jnp.ndarray] = None,
        syndrome: jnp.ndarray = None,
    ):
        hx, hz, lx, lz = parity_info
        m, n = hx.shape
        if error is None:
            error = (
                jnp.zeros(shape=n, dtype=jnp.bool),
                jnp.zeros(shape=n, dtype=jnp.bool)
            )
            logicals = None
        if syndrome is None:
            syndrome, logicals = self.syndrome(error, parity_info)

        plt.figure()
        # Plot the X-connections
        idx_syndrome, idx_data = jnp.where(
            jnp.logical_and(hx == 0, hz == 1))
        xs, ys = jnp.ravel(jnp.column_stack((
            self.data_qubit_loc[idx_data],
            self.syndrome_qubit_loc[idx_syndrome],
            # For gaps between line segments
            jnp.zeros(shape=(idx_data.shape[0], 2))*jnp.inf)
        )).reshape(-1, 2).T
        plt.plot(xs, ys, label='X', color='#FF0000')
        # Plot the Y-connections
        idx_syndrome, idx_data = jnp.where(
            jnp.logical_and(hx == 1, hz == 1))
        xs, ys = jnp.ravel(jnp.column_stack((
            self.data_qubit_loc[idx_data],
            self.syndrome_qubit_loc[idx_syndrome],
            # For gaps between line segments
            jnp.zeros(shape=(idx_data.shape[0], 2))*jnp.inf)
        )).reshape(-1, 2).T
        plt.plot(xs, ys, label='Y', color='#00FF00')
        # Plot the Z-connections
        idx_syndrome, idx_data = jnp.where(
            jnp.logical_and(hx == 1, hz == 0))
        xs, ys = jnp.ravel(jnp.column_stack((
            self.data_qubit_loc[idx_data],
            self.syndrome_qubit_loc[idx_syndrome],
            # For gaps between line segments
            jnp.zeros(shape=(idx_data.shape[0], 2))*jnp.inf)
        )).reshape(-1, 2).T
        plt.plot(xs, ys, label='Z', color='#0000FF')
        # Plot the data qubits
        plt.scatter(
            x=self.data_qubit_loc[:, 0],
            y=self.data_qubit_loc[:, 1],
            c=error[0] + 2*error[1],  # I X Z Y
            s=100,
            cmap=ListedColormap(['black', 'red', 'blue', 'green']),
            label='data qubits',
            vmin=-.5,
            vmax=3.5,
            zorder=2
        )
        # Plot the syndrome qubits
        plt.scatter(
            x=self.syndrome_qubit_loc[:, 0],
            y=self.syndrome_qubit_loc[:, 1],
            c=syndrome,
            s=syndrome*100+10,
            cmap=ListedColormap(['black', 'gray']),
            label='syndrome qubits',
            vmin=-.5,
            vmax=1.5,
            zorder=2
        )
        # Make legend
        # plt.legend()
        plt.gca().set_aspect('equal')
        if logicals is not None:
            plt.title(f"Logicals parity: {logicals}")
        plt.show()


def surface_code(L: int) -> QEC:
    """
    The distance of the code must be an odd number.

    Returns: QEC object that represents the surface code
    """
    if L % 2 == 0:
        raise ValueError(
            f"The distance of the surface code must be an odd number but got {L=}")
    # Create the qubit and syndrome locations
    data_qubit_loc = jnp.column_stack((
        jnp.arange(L**2, dtype=jnp.float32)[:, None] % L,
        jnp.arange(L**2, dtype=jnp.float32)[:, None] // L % L,
    ))
    syndrome_qubit_loc = jnp.column_stack((
        jnp.arange((L+1)**2, dtype=jnp.float32)[:, None] % (L+1) - .5,
        jnp.arange((L+1)**2, dtype=jnp.float32)[:, None] // (L+1) % (L+1) - .5,
    ))
    # Create the logical x and z operators
    logical_x = jnp.zeros(
        shape=(2, L**2),
        dtype=jnp.int32,
    ).at[0, :L].set(1)
    logical_z = jnp.zeros(
        shape=(2, L**2),
        dtype=jnp.int32,
    ).at[1, ::L].set(1)
    # Create the parity check matricies
    """
    Explanation of jnp.arange(2*d**2)//2 + jnp.arange(2*d**2)//2//d + jnp.array([0,d+2,d+1,1])[jnp.arange(2*d**2)%4]
    jnp.arange(2*d**2)//2: Gives each data qubit two connections to work with initially set to the bottom left stabilizer
    jnp.arange(2*d**2)//2//d: Accounts for the extra syndrome qubit in each row compared to data qubit
    jnp.array([0,d+2,d+1,1])[jnp.arange(2*d**2)%4]: Remaps the connections to the correct syndrome qubits (fist two terms for even data qubits, second two terms for odd data qubits)
    """
    hx = jnp.zeros(
        shape=(syndrome_qubit_loc.shape[0], data_qubit_loc.shape[0]),
        dtype=jnp.int32
    ).at[(
        jnp.arange(2*L**2)//2 + jnp.arange(2*L**2)//2//L +
        jnp.array([1, L+1, L+2, 0])[jnp.arange(2*L**2) % 4],
        jnp.arange(2*L**2)//2,
    )].set(1)
    hz = jnp.zeros(
        shape=(syndrome_qubit_loc.shape[0], data_qubit_loc.shape[0]),
        dtype=jnp.int32
    ).at[(
        jnp.arange(2*L**2)//2 + jnp.arange(2*L**2)//2//L +
        jnp.array([0, L+2, L+1, 1])[jnp.arange(2*L**2) % 4],
        jnp.arange(2*L**2)//2,
    )].set(1)
    # Create mask to remove excess syndrome qubits
    mask = jnp.zeros(shape=(L+1, L+1), dtype=jnp.bool)
    # Mask for the x-stabilizers
    mask = mask.at[1:-1:2, ::2].set(True).at[2:-1:2, 1::2].set(True)
    # Mask for the z-stabilizers
    mask = mask.at[1::2, 1:-1:2].set(True).at[::2, 2:-1:2].set(True)
    # Remove excess syndrome qubits
    syndrome_qubit_loc = syndrome_qubit_loc[jnp.ravel(mask)]
    hx = hx[jnp.ravel(mask), :]
    hz = hz[jnp.ravel(mask), :]
    return QEC(
        data_qubit_loc,
        syndrome_qubit_loc,
        parity_check_matrix=jnp.append(hx[None, :, :], hz[None, :, :], axis=0),
        logical_parity_matrix=jnp.append(
            logical_x[None, :, :], logical_z[None, :, :], axis=0),
        mask=mask,
    )


def get_deformation_image(
    code: QEC,
    deformation: jnp.ndarray,
    error_probs: jnp.ndarray,
):
    """
    Only works for the surface code gotten by calling `surface_code(L)`

    Returns a four channel image with information about the weight of the error that could trigger that stabilizer from each of the four directions (NV, NE, SV and SE)
    """
    def deform_stab_idx(
        stab: jnp.ndarray,
        deformation: jnp.ndarray,
    ):
        transformations_idx = jnp.array([
            [0, 1, 2, 3],  # I
            [0, 2, 1, 3],  # X-Y
            [0, 1, 3, 2],  # Y-Z
            [0, 3, 2, 1],  # X-Z
            [0, 2, 3, 1],  # X-Y-Z
            [0, 3, 1, 2],  # X-Z-Y
        ])
        transformation_table = transformations_idx[deformation]
        return transformation_table[(
            jnp.arange(stab.shape[0]),
            stab
        )]

    n, m = jnp.array(code.mask.shape) - 1
    graph_idx = jnp.arange(n*m) % 2
    stab_a = deform_stab_idx(
        jnp.where(graph_idx == 0, 3, 1), deformation).reshape(n, m)
    stab_b = deform_stab_idx(
        jnp.where(graph_idx == 0, 1, 3), deformation).reshape(n, m)

    stab_dirs = jnp.stack((
        jnp.pad(stab_a, pad_width=((0, 1), (1, 0))),  # NV
        jnp.pad(stab_b, pad_width=((0, 1), (0, 1))),  # NE
        jnp.pad(stab_b, pad_width=((1, 0), (1, 0))),  # SV
        jnp.pad(stab_a, pad_width=((1, 0), (0, 1))),  # SE
    )) * code.mask[None, :, :]

    weights = jnp.log(error_probs) / jnp.log(error_probs.max())
    stab_weights = jnp.array([
        0,
        weights[1] + weights[2],
        weights[0] + weights[2],
        weights[0] + weights[1],
    ])
    weight_dirs = stab_weights[stab_dirs]

    return weight_dirs

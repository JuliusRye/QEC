import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, jit
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
        self.hx, self.hz = parity_check_matrix
        self.lx, self.lz = logical_parity_matrix
        self.deformation = jnp.zeros(shape=self.hx.shape[1], dtype=jnp.int32)
        self.mask = mask

    @partial(jit, static_argnames=("self"))
    def error(
        self,
        key,
        probabilities: jnp.ndarray,
    ) -> tuple[jnp.ndarray]:
        px, py, pz = probabilities
        n = self.hx.shape[1]
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
        error: (jnp.ndarray)
    ) -> jnp.ndarray:
        # Calculate syndrome
        parity_x = jnp.matmul(self.hx, error[0])
        parity_z = jnp.matmul(self.hz, error[1])
        syndrome = (parity_x + parity_z) % 2
        # Calculate logicals
        parity_x = jnp.matmul(self.lx, error[0])
        parity_z = jnp.matmul(self.lz, error[1])
        logicals = (parity_x + parity_z) % 2
        return syndrome, logicals

    @partial(jit, static_argnames=("self"))
    def syndrome_img(
        self,
        error: (jnp.ndarray)
    ) -> jnp.ndarray:
        """
        Creates an image of the syndrome from the error with:

        -1: Detected a syndrome here

        0: No syndrome measurement at this location

        1: Detected no syndrome
        """
        syndrome, logicals = self.syndrome(error)
        syndrome = syndrome.astype(jnp.int32)
        img = jnp.zeros_like(
            self.mask,
            dtype=jnp.float32
        ).at[self.mask].set(1 - 2*syndrome)
        return img, logicals

    def show(
        self,
        error: tuple[jnp.ndarray] = None,
        syndrome: jnp.ndarray = None,
    ):
        m, n = self.hx.shape
        if error is None:
            error = (
                jnp.zeros(shape=n, dtype=jnp.bool),
                jnp.zeros(shape=n, dtype=jnp.bool)
            )
            logicals = None
        if syndrome is None:
            syndrome, logicals = self.syndrome(error)

        plt.figure()
        # Plot the X-connections
        idx_syndrome, idx_data = jnp.where(
            jnp.logical_and(self.hx == 0, self.hz == 1))
        xs, ys = jnp.ravel(jnp.column_stack((
            self.data_qubit_loc[idx_data],
            self.syndrome_qubit_loc[idx_syndrome],
            # For gaps between line segments
            jnp.zeros(shape=(idx_data.shape[0], 2))*jnp.inf)
        )).reshape(-1, 2).T
        plt.plot(xs, ys, label='X', color='#FF0000')
        # Plot the Y-connections
        idx_syndrome, idx_data = jnp.where(
            jnp.logical_and(self.hx == 1, self.hz == 1))
        xs, ys = jnp.ravel(jnp.column_stack((
            self.data_qubit_loc[idx_data],
            self.syndrome_qubit_loc[idx_syndrome],
            # For gaps between line segments
            jnp.zeros(shape=(idx_data.shape[0], 2))*jnp.inf)
        )).reshape(-1, 2).T
        plt.plot(xs, ys, label='Y', color='#00FF00')
        # Plot the Z-connections
        idx_syndrome, idx_data = jnp.where(
            jnp.logical_and(self.hx == 1, self.hz == 0))
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

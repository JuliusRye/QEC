import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, jit, vmap
from functools import partial
from matplotlib.colors import ListedColormap
import stim
from matplotlib.colors import ListedColormap


deformation_cmap = ListedColormap([
    "#FFFFFF",  # I
    "#88C946",  # X <-> Y
    "#C43119",  # Y <-> Z
    "#00ADE7",  # X <-> Z
    "#833794",  # X -> Z -> Y -> X
    "#FFB94C",  # X -> Y -> Z -> X
])
deformation_cmap.set_under("k", alpha=0)
deformation_cmap.set_over("k", alpha=0)


class QEC:

    def __init__(
        self,
        data_qubit_loc: jnp.ndarray,
        syndrome_qubit_loc: jnp.ndarray,
        parity_check_matrix: jnp.ndarray,
        logical_parity_matrix: jnp.ndarray,
        act_order: jnp.ndarray,
    ):
        """
        data_qubit_loc: shape=(n,2) array of 2d location of n data qubits
        syndrome_qubit_loc: shape=(s,2) array of 2d location of s syndrome qubits
        parity_check_matrix: A matrix of shape=(2,s,n) of parity check X and Z for the s syndrome measurements
        logical_parity_matrix: A matrix of shape=(2,l,n) of parity check X and Z for the l logical observables
        act_order: A matrix of shape=(s,n) of int denoting the timestep to apply any stabilizer between syndrome qubit s and data qubit n
        mask: A 2d array with true false values to map the syndrome onto (Number of true values should equal number of syndrome qubits)
        """
        self.data_qubit_loc = data_qubit_loc
        self.syndrome_qubit_loc = syndrome_qubit_loc
        self.act_order = act_order

        self.num_data_qubits = data_qubit_loc.shape[0]
        self.num_syndrome_qubits = syndrome_qubit_loc.shape[0]

        self.hx_original, self.hz_original = parity_check_matrix[0], parity_check_matrix[1]
        self.lx_original, self.lz_original = logical_parity_matrix[0], logical_parity_matrix[1]

        self.deformation = jnp.zeros(
            shape=self.num_data_qubits, dtype=jnp.int32)

    def random_deformation(
        self,
        key,
        allowed_deformations: jnp.ndarray
    ):
        """
        Creates a random deformation.

        allowed_deformations: Array of the deformation idx's that is allowed to be used

        returns: tuple of (deformation, key)
        """
        return self._random_deformation(key, allowed_deformations)

    @partial(jit, static_argnames=("self"))
    def _random_deformation(
        self,
        key,
        allowed_deformations: jnp.ndarray
    ):
        subkey, key = random.split(key)
        deformation = allowed_deformations[random.randint(
            subkey,
            shape=self.num_data_qubits,
            minval=0,
            maxval=allowed_deformations.shape[0]
        )]
        return deformation, key

    def deformation_parity_info(
        self,
        D: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

        transformations = jnp.array([
            [[1, 0], [0, 1]],  # I
            [[1, 1], [0, 1]],  # X-Y
            [[1, 0], [1, 1]],  # Y-Z
            [[0, 1], [1, 0]],  # X-Z
            [[1, 1], [1, 0]],  # X-Z-Y
            [[0, 1], [1, 1]],  # X-Y-Z
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

        hx = A_prime[:self.num_syndrome_qubits]
        lx = A_prime[self.num_syndrome_qubits:]

        hz = B_prime[:self.num_syndrome_qubits]
        lz = B_prime[self.num_syndrome_qubits:]

        parity_info = (hx, hz, lx, lz)
        return parity_info

    def error(
        self,
        key,
        probabilities: jnp.ndarray,
    ) -> tuple[jnp.ndarray]:
        """
        #### Jit optimized function!

        Generate data qubit errors.

        key: jax random key for generating random numbers

        probabilities: Pauli errors probabilities for [X, Y, Z] errors respectively

        returns: An array of X and Z error locations of shape (2, num_data_qubits)
        """
        return self._error(key, probabilities)

    @partial(jit, static_argnames=("self"))
    def _error(
        self,
        key,
        probabilities: jnp.ndarray
    ) -> tuple[jnp.ndarray]:
        px, py, pz = probabilities
        rv = random.uniform(key, shape=self.num_data_qubits)
        error_x = rv < (px + py)
        error_z = jnp.logical_and(
            rv > px,
            rv < px + py + pz
        )
        return jnp.column_stack((
            error_x,
            error_z
        )).T

    def syndrome(
        self,
        error: jnp.ndarray,
        parity_info: tuple[jnp.ndarray],
    ) -> tuple[jnp.ndarray]:
        """
        #### Jit optimized function!

        The syndromes created by a given error on the codes data qubits.

        error: An array of X and Z error locations of shape (2, num_data_qubits)

        parity_info: A tuple (hx, hz, lx, lz) of the stabilizer X and Z and logical X and Z parity check matrix

        returns: tuple of the syndrome array and logical array
        """
        return self._syndrome(error, parity_info)

    @partial(jit, static_argnames=("self"))
    def _syndrome(
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

    def show(
        self,
        parity_info: tuple[jnp.ndarray],
        error: tuple[jnp.ndarray] = None,
    ):
        hx, hz, lx, lz = parity_info
        m, n = hx.shape
        if error is None:
            error = jnp.zeros(shape=(2,n), dtype=jnp.bool)
            logicals = None
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

    def show_latex_code(
        self,
        parity_info,
        deformation,
        error: jnp.ndarray = None,
    ):
        hx, hz, lx, lz = parity_info
        m, n = hx.shape
        if error is None:
            error = jnp.zeros(shape=(2,n), dtype=jnp.bool)
            logicals = None
        syndrome, logicals = self.syndrome(error, parity_info)
        xerr, zerr = error

        colors = ["White", "LimeGreen", "BrickRed",
                  "Cerulean", "Fuchsia", "Dandelion"]
        err_color = [ # Order I, X, Z, Y
            None,
            "red!80",
            "Cyan",
            "green!80",
        ]

        stab_x = jnp.where(jnp.logical_and(hx == 0, hz == 1))
        stab_y = jnp.where(jnp.logical_and(hx == 1, hz == 1))
        stab_z = jnp.where(jnp.logical_and(hx == 1, hz == 0))

        latex_code = ""

        latex_code += f"\n% Draw the data qubits\n"
        for i, (x, y) in enumerate(self.data_qubit_loc):
            latex_code += f"\\node[draw, circle, fill={colors[deformation[i]]}, line width=.5mm, minimum size=5mm] (D{i}) at ({x*2},{y*2}) {{}};\n"
            if xerr[i]+2*zerr[i] > 0:
                latex_code += f"\\node[draw, star, fill={err_color[xerr[i]+2*zerr[i]]}, line width=.2mm, minimum size=1mm, inner sep=.6mm] () at ({x*2},{y*2}) {{}};\n"

        latex_code += f"\n% Draw the syndrome qubits\n"
        for i, (x, y) in enumerate(self.syndrome_qubit_loc):
            latex_code += f"\\node[draw, circle, fill={'gray' if syndrome[i] else 'black'}, line width=.5mm, minimum size=2mm] (S{i}) at ({x*2},{y*2}) {{}};\n"

        latex_code += f"\n% Draw the x stabilizers\n"
        for i, j in zip(*stab_x):
            if (error[:,j] == jnp.array([0,1])).all() or (error[:,j] == jnp.array([1,1])).all():
                latex_code += f"\\draw[black, line width=1.3mm] (S{i}) -- (D{j});\n"
            latex_code += f"\\draw[red!80, line width=.3mm] (S{i}) -- (D{j});\n"

        latex_code += f"\n% Draw the y stabilizers\n"
        for i, j in zip(*stab_y):
            if (error[:,j] == jnp.array([0,1])).all() or (error[:,j] == jnp.array([1,0])).all():
                latex_code += f"\\draw[black, line width=1.3mm] (S{i}) -- (D{j});\n"
            latex_code += f"\\draw[green!80, line width=.3mm] (S{i}) -- (D{j});\n"

        latex_code += f"\n% Draw the z stabilizers\n"
        for i, j in zip(*stab_z):
            if (error[:,j] == jnp.array([1,0])).all() or (error[:,j] == jnp.array([1,1])).all():
                latex_code += f"\\draw[black, line width=1.3mm] (S{i}) -- (D{j});\n"
            latex_code += f"\\draw[Cyan, line width=.3mm] (S{i}) -- (D{j});\n"

        return latex_code

    def to_stim(
        self,
        parity_info: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        rounds=1,
        # Operational times
        idelig_T1_us: float = 0.0,
        idelig_T2_us: float = 0.0,
        single_qubit_gate_time_us: float = 0.05,
        two_qubit_gate_time_us: float = 0.4,
        meas_time_us: float = 1.0,
        # Error rates
        single_qubit_error_rate: float = 0.0,
        two_qubit_error_rate: float = 0.0,
        meas_error_rate: float = 0.0,
        reset_error_rate: float = 0.0,
        # Initialization
        init_basis: str = 'Z',
    ):
        """
        rounds: how many times each stabilizer should be measured (usually the code distance)
        """
        hx, hz, lx, lz = parity_info
        nsq, ndq = hx.shape

        def px(t): return (1 - jnp.exp(-t/idelig_T1_us)) / 4
        def py(t): return (1 - jnp.exp(-t/idelig_T1_us)) / 4
        def pz(t): return (1 + jnp.exp(-t/idelig_T1_us) -
                           2*jnp.exp(-t/idelig_T2_us)) / 4

        # Create the quantum circuit for a single QEC round
        circ_init = stim.Circuit()
        circ_end = stim.Circuit()
        logi_to_z_basis = stim.Circuit()
        qec_round = stim.Circuit()
        stab_order = [
            stim.Circuit(),  # SW
            stim.Circuit(),  # SE
            stim.Circuit(),  # NW
            stim.Circuit(),  # NE
        ]
        qubits = jnp.arange(ndq+nsq)
        data_qubits = qubits[:ndq]
        synd_qubits = qubits[ndq:]

        # Assign coordinates to data qubits
        for i, coord in enumerate(self.data_qubit_loc):
            circ_init.append_operation(
                "QUBIT_COORDS", [data_qubits[i]], [coord[0], -coord[1]])

        # Assign coordinates to syndrome qubits
        for i, coord in enumerate(self.syndrome_qubit_loc):
            circ_init.append_operation(
                "QUBIT_COORDS", [synd_qubits[i]], [coord[0], -coord[1]])
        circ_init.append_operation("R", qubits)
        if reset_error_rate > 0:
            circ_init.append_operation(
                "X_ERROR", qubits, reset_error_rate)

        circ_end.append_operation("TICK")
        if meas_error_rate > 0:
            circ_end.append_operation(
                "X_ERROR", data_qubits, meas_error_rate)
        circ_end.append_operation("M", data_qubits)

        match init_basis.upper():
            case "Z":
                logical_stabilizer = lz
            case "X":
                logical_stabilizer = lx
            case "Y":
                logical_stabilizer = (lx + lz) % 2
        stab_x = jnp.where(jnp.all(logical_stabilizer ==
                           jnp.array([[1], [0]]), axis=0))[0]
        stab_y = jnp.where(jnp.all(logical_stabilizer ==
                           jnp.array([[1], [1]]), axis=0))[0]
        stab_z = jnp.where(jnp.all(logical_stabilizer ==
                           jnp.array([[0], [1]]), axis=0))[0]
        logi_to_z_basis.append_operation("H_XZ", data_qubits[stab_x])
        logi_to_z_basis.append_operation("H_YZ", data_qubits[stab_y])
        logi_to_z_basis.append_operation("I", data_qubits[stab_z])
        if single_qubit_error_rate > 0:
            logi_to_z_basis.append_operation(
                "DEPOLARIZE1", data_qubits, single_qubit_error_rate)

        # Apply parity checks based on hx and hz
        used_qubits = jnp.zeros((4, qubits.shape[0]), dtype=jnp.bool)
        for d in range(ndq):
            for s in range(nsq):
                order_idx = self.act_order[s, d]
                match (hx[s, d], hz[s, d]):
                    case (0, 1):  # X
                        stab_order[order_idx].append_operation(
                            "CX",
                            [synd_qubits[s], data_qubits[d]]
                        )
                    case (1, 1):  # Y
                        stab_order[order_idx].append_operation(
                            "CY",
                            [synd_qubits[s], data_qubits[d]]
                        )
                    case (1, 0):  # Z
                        stab_order[order_idx].append_operation(
                            "CZ",
                            [synd_qubits[s], data_qubits[d]]
                        )
                if hx[s, d] + hz[s, d] > 0:
                    if two_qubit_error_rate > 0:
                        stab_order[order_idx].append_operation(
                            "DEPOLARIZE2",
                            [synd_qubits[s], data_qubits[d]],
                            two_qubit_error_rate
                        )
                    used_qubits = used_qubits.at[order_idx, synd_qubits[s]].set(
                        True)
                    used_qubits = used_qubits.at[order_idx, data_qubits[d]].set(
                        True)
        if idelig_T1_us > 0 or idelig_T2_us:
            for i in range(len(stab_order)):
                t = two_qubit_gate_time_us
                stab_order[i].append_operation("PAULI_CHANNEL_1", jnp.where(
                    used_qubits[i] == False)[0], [px(t), py(t), pz(t)])

        qec_round.append_operation("TICK")
        if idelig_T1_us > 0 or idelig_T2_us:
            # Time = 2 H-gates + 1 reset + 1 measurement
            t = 3*single_qubit_gate_time_us + meas_time_us
            qec_round.append_operation(
                "PAULI_CHANNEL_1", data_qubits, [px(t), py(t), pz(t)])
        qec_round.append_operation("H", synd_qubits)
        if single_qubit_error_rate > 0:
            qec_round.append_operation(
                "DEPOLARIZE1", synd_qubits, single_qubit_error_rate)
        qec_round.append_operation("TICK")
        qec_round += stab_order[0]  # SW
        qec_round.append_operation("TICK")
        qec_round += stab_order[1]  # SE
        qec_round.append_operation("TICK")
        qec_round += stab_order[2]  # NW
        qec_round.append_operation("TICK")
        qec_round += stab_order[3]  # NE
        qec_round.append_operation("TICK")
        qec_round.append_operation("H", synd_qubits)
        if single_qubit_error_rate > 0:
            qec_round.append_operation(
                "DEPOLARIZE1", synd_qubits, single_qubit_error_rate)
        qec_round.append_operation("TICK")
        if meas_error_rate > 0:
            qec_round.append_operation(
                "X_ERROR", synd_qubits, meas_error_rate)
        qec_round.append_operation("MR", synd_qubits)
        if reset_error_rate > 0:
            qec_round.append_operation(
                "X_ERROR", synd_qubits, reset_error_rate)

        detectors_first = stim.Circuit()
        detectors = stim.Circuit()
        for s in range(nsq):
            detectors_first.append_operation(
                "DETECTOR", [stim.target_rec(s-nsq)])
            detectors.append_operation(
                "DETECTOR", [stim.target_rec(s-nsq), stim.target_rec(s-2*nsq)])

        return circ_init + logi_to_z_basis + qec_round + (qec_round+detectors)*rounds + logi_to_z_basis + circ_end


class SurfaceCode(QEC):

    def __init__(
        self,
        L: int,
    ):
        """
        Creates the distance L rotated surface code.

        L: Code distance
        """
        self.L = L
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
            jnp.arange(
                (L+1)**2, dtype=jnp.float32)[:, None] // (L+1) % (L+1) - .5,
        ))
        # Create the logical x and z operators
        logical_x = jnp.zeros(
            shape=(2, L**2),
            dtype=jnp.int32,
        ).at[0, :].set(1)
        logical_z = jnp.zeros(
            shape=(2, L**2),
            dtype=jnp.int32,
        ).at[1, :].set(1)
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
        # Pick the order to apply the stabilizers parts in
        act_order = -jnp.ones(((L+1)**2, L**2), dtype=jnp.int32)
        diagstep_i = jnp.arange(L**2)+jnp.arange(L**2)//L
        diagstep_j = jnp.arange(L**2)
        act_order = act_order.at[diagstep_i+L+1, diagstep_j].set(0)  # NW
        # NE & SW alternating
        act_order = act_order.at[diagstep_i+0,
                                 diagstep_j].set((jnp.arange(L**2)+1) % 2 + 1)
        # SW & NE alternating
        act_order = act_order.at[diagstep_i+L+2,
                                 diagstep_j].set((jnp.arange(L**2)+0) % 2 + 1)
        act_order = act_order.at[diagstep_i+1, diagstep_j].set(3)  # SE
        # for idx, stab in zip(act_order, hx+hz):
        #     print(" ".join(" " if i == 0 else "_" for i in stab))
        #     print(" ".join(" " if i == -1 else str(i) for i in idx))
        # Create mask to remove excess syndrome qubits
        self.mask = jnp.zeros(shape=(L+1, L+1), dtype=jnp.bool)
        # Mask for the x-stabilizers
        self.mask = self.mask.at[1:-1:2,
                                 ::2].set(True).at[2:-1:2, 1::2].set(True)
        # Mask for the z-stabilizers
        self.mask = self.mask.at[1::2, 1:-
                                 1:2].set(True).at[::2, 2:-1:2].set(True)
        # Remove excess syndrome qubits
        syndrome_qubit_loc = syndrome_qubit_loc[jnp.ravel(self.mask)]
        hx = hx[jnp.ravel(self.mask), :]
        hz = hz[jnp.ravel(self.mask), :]
        act_order = act_order[jnp.ravel(self.mask), :]
        # Initialize the quantum error correction code
        super().__init__(
            data_qubit_loc,
            syndrome_qubit_loc,
            parity_check_matrix=jnp.append(
                hx[None, :, :], hz[None, :, :], axis=0),
            logical_parity_matrix=jnp.append(
                logical_x[None, :, :], logical_z[None, :, :], axis=0),
            act_order=act_order,
        )

    def syndrome_img(
        self,
        error: (jnp.ndarray),
        parity_info: tuple[jnp.ndarray],
    ) -> tuple[jnp.ndarray]:
        """
        #### Jit optimized function!

        Same functionality as the syndrome function but aranged in a 2d image with:

        1: Detected no syndrome;
        -1: Detected a syndrome;
        0: No syndrome measurement;

        error: An array of X and Z error locations of shape (2, num_data_qubits)

        parity_info: A tuple (hx, hz, lx, lz) of the stabilizer X and Z and logical X and Z parity check matrix

        returns: tuple of the syndrome image matrix and logical array
        """
        return self._syndrome_img(error, parity_info)

    @partial(jit, static_argnames=("self"))
    def _syndrome_img(
        self,
        error: (jnp.ndarray),
        parity_info: tuple[jnp.ndarray],
    ) -> tuple[jnp.ndarray]:
        syndrome, logicals = self.syndrome(error, parity_info)
        syndrome = syndrome.astype(jnp.int32)
        img = jnp.zeros_like(
            self.mask,
            dtype=jnp.float32
        ).at[self.mask].set(1 - 2*syndrome)
        return img, logicals

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
        return self._deformation_image(deformation)

    @partial(jit, static_argnames=("self"))
    def _deformation_image(
        self,
        deformation: jnp.ndarray,
    ):
        img_deformation = jnp.eye(6, dtype=jnp.float32)[
            deformation.reshape((self.L, self.L))
        ].transpose(2, 0, 1)
        return img_deformation
    
    def show_latex_code(
        self, 
        parity_info: jnp.ndarray, 
        deformation: jnp.ndarray,
        error: jnp.ndarray = None,
    ):
        
        latex_code = "% Draw the plaquettes\n"

        # Background plaquets
        for i in range(self.L-1):
            for j in range(self.L-1):
                x1, x2 = 2*i, 2*(i+1)
                y1, y2 = 2*j, 2*(j+1)
                latex_code += f"\\filldraw[fill=black!{5 if (i+j)%2 else 10}, draw=none] ({x1},{y1}) -- ({x2},{y1}) -- ({x2},{y2}) -- ({x1},{y2}) -- cycle;\n"
        for i in range((self.L-1)//2):
            dim = 2*(self.L-1)
            latex_code += f"\\filldraw[fill=black!5, draw=none] ({0},{4*i}) -- ({0},{4*i+2}) -- ({-1},{4*i+1}) -- cycle;\n"
            latex_code += f"\\filldraw[fill=black!5, draw=none] ({dim},{4*i+2}) -- ({dim},{4*i+4}) -- ({dim+1},{4*i+3}) -- cycle;\n"
            latex_code += f"\\filldraw[fill=black!10, draw=none] ({4*i+2},{0}) -- ({4*i+4},{0}) -- ({4*i+3},{-1}) -- cycle;\n"
            latex_code += f"\\filldraw[fill=black!10, draw=none] ({4*i},{dim}) -- ({4*i+2},{dim}) -- ({4*i+1},{dim+1}) -- cycle;\n"

        return latex_code + super().show_latex_code(parity_info, deformation, error)

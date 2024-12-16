import jax
import jax.numpy as jnp

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, AncillaRegister
from qiskit.quantum_info import StabilizerState, Statevector

from icecream import ic  # For debugging

deformations = {  # The different Clifford deformations (neglecting the phase)
    'I': {'X': 'X', 'Y': 'Y', 'Z': 'Z', 'color': 'black'},
    'XZ': {'X': 'Z', 'Y': 'Y', 'Z': 'X', 'color': '#43A2D9'},
    'YZ': {'X': 'X', 'Y': 'Z', 'Z': 'Y', 'color': '#FF604B'},
    'XY': {'X': 'Y', 'Y': 'X', 'Z': 'Z', 'color': '#43D972'},
    'XYZ': {'X': 'Y', 'Y': 'Z', 'Z': 'X', 'color': 'gray'},
    'XZY': {'X': 'Z', 'Y': 'X', 'Z': 'Y', 'color': 'gray'}
}


def state_from_stabilizers(stabilizer_list: list[str], allow_underconstrained=False, remove_global_phase=True) -> Statevector:
    stabilizers = StabilizerState.from_stabilizer_list(
        stabilizer_list, allow_underconstrained=allow_underconstrained)
    statevector = Statevector.from_label(
        '0'*stabilizers.num_qubits).evolve(stabilizers)
    if remove_global_phase == True:
        idx = (statevector.data != 0).argmax(axis=0)  # First nonzero element
        phase_0 = np.angle(statevector.data[idx])
        statevector = statevector * np.exp(-1j * phase_0)
    return statevector


def jax_get_syndromes(Mx: jnp.ndarray, My: jnp.ndarray, Mz: jnp.ndarray, err: jnp.ndarray) -> jnp.ndarray:
    return ((
        jnp.matmul(Mx, err == 1) +
        jnp.matmul(My, err == 2) +
        jnp.matmul(Mz, err == 3)
    ) % 2).astype(int)


def jax_get_syndromes_batch(Mx: jnp.ndarray, My: jnp.ndarray, Mz: jnp.ndarray, err: jnp.ndarray) -> jnp.ndarray:
    batch_get_syndromes = jax.vmap(
        jax_get_syndromes,
        in_axes=(None, None, None, 0),
        out_axes=0)
    return batch_get_syndromes(Mx, My, Mz, err)


def jax_create_error(px: float, py: float, pz: float, size: int, key) -> jnp.ndarray:
    rand = jax.random.uniform(key, shape=(size))
    return jnp.where(rand < py+pz, jnp.where(rand < pz, 3, 2), jnp.where(rand < px+py+pz, 1, 0))


def jax_create_error_batch(px: float, py: float, pz: float, size: int, batch_size: int, key) -> jnp.ndarray:
    keys = jax.random.split(key, batch_size)
    batch_created_errors = jax.vmap(
        jax_create_error,
        in_axes=(None, None, None, None, 0),
        out_axes=0)
    return batch_created_errors(px, py, pz, size, keys)


class Qubit():

    def __init__(self, location: tuple, acted_on_by: list = [], name: str = None, connections=True) -> None:
        """
        location: tuple (x,y,z)
        acted_on_by: list of dicts {index_of_stabilizer_qubit, pauli_action}
        """
        self.location = location
        self.acted_on_by = acted_on_by
        self.interior_color = 'white'
        self.qubit = None
        self.index = None
        self.name = name
        self.connections = connections

    def deform(self, deformation: dict[str, str]) -> None:
        for i, (index, pauli) in enumerate(self.acted_on_by):
            self.acted_on_by[i] = [index, deformation[pauli]]
        self.interior_color = deformation['color']


class QEC():

    def _size(self, qubits: list[list | Qubit]) -> int:
        """
        Returns the number of qubits in a nested list of qubits excluding any hidden qubits
        """
        return sum(
            self._size(qubit) if isinstance(qubit, list) else
            (0 if qubit.location[0] is None else 1)
            for qubit in qubits
        )

    def _flatten(self, qubits: list[list | Qubit]) -> list[Qubit]:
        """
        Flattens a nested list of qubits and excludes any hidden qubits
        """
        result = []
        for qubit in qubits:
            if isinstance(qubit, list):  # Check if the current element is a list
                # Recursively flatten the nested list
                result.extend(self._flatten(qubit))
            else:
                if qubit.location[0] == None:
                    continue
                result.append(qubit)  # Add non-list elements to the result
        return result

    def _from_index(self, qubits: list[list | Qubit], index: list[int]) -> Qubit:
        """
        Returns the qubit from a nested list of qubits
        """
        return qubits[index[0]] if len(index) == 1 else self._from_index(qubits[index[0]], index[1:])

    def __init__(self, data: dict[str, Qubit]) -> None:
        """
        data["data_qubits"]: iterable of DataQubits
        """
        self.data = data
        # Define Pauli colors
        self.stabilizer_colors = {
            'X': '#aa0000',
            'Y': '#00aa00',
            'Z': '#0000aa',
        }
        self.error_colors = [
            'black',  # Identity
            'red',    # Pauli X
            'green',  # Pauli Y
            'blue'    # Pauli Z
        ]

    def deform(self, qubit_index: list, deformation: dict[str, str]) -> None:
        qubit: Qubit = self._from_index(self.data["data_qubits"], qubit_index)
        qubit.deform(deformation)

    def transformation_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the transformation matrix ``Mx, My, Mz`` for the Pauli X, Y and Z errors respectively.

        Transforms a flattend list of a given Pauli error (Ex, Ey or Ez) into the resulting syndromes.
        The final syndrome will be ``(Mx*Ex + My*Ey + Mz*Ez) mod 2``
        """
        # Flatten the qubits and index them in order
        for idx, qubit in enumerate(self._flatten(self.data["data_qubits"])):
            qubit.index = idx
        for idx, qubit in enumerate(self._flatten(self.data["synd_qubits"])):
            qubit.index = idx
        # Calculate the transformation matrices
        Mx = np.zeros(
            (self._size(self.data["synd_qubits"]), self._size(self.data["data_qubits"])))
        My = Mx.copy()
        Mz = Mx.copy()
        for idx_a, qubit in enumerate(self._flatten(self.data["data_qubits"])):
            for (index, pauli) in qubit.acted_on_by:
                idx_b = self._from_index(self.data["synd_qubits"], index).index
                if idx_b is None:
                    continue
                match pauli:
                    case 'X':
                        Mz[idx_b, idx_a] = 1
                        My[idx_b, idx_a] = 1
                    case 'Y':
                        Mx[idx_b, idx_a] = 1
                        Mz[idx_b, idx_a] = 1
                    case 'Z':
                        Mx[idx_b, idx_a] = 1
                        My[idx_b, idx_a] = 1
                    case _:
                        print(pauli)
        return Mx, My, Mz

    def show(self, axis: Axes = None, elev=60, azim=0, roll=0, errors=None, marker_size=15, title: str = "") -> Axes:
        """
        Makes a 3d plot of the data qubits, measurement qubits and the pauli connections between them
        """
        # Create a figure axis if no axis was passed
        if axis is None:
            fig = plt.figure(dpi=300)
            axis = fig.add_subplot(projection='3d')
        # Plot qubits and stabilizers
        if errors is None:
            # Set data qubits errors to identity (value: 0)
            errors = jnp.zeros(self._size(self.data["data_qubits"]))
        # Plot the data qubits
        for qubit, error in zip(self._flatten(self.data["data_qubits"]), errors):
            ax, ay, az = qubit.location
            axis.plot(ax, ay, az, '.', ms=marker_size, c='black',
                      mfc=self.error_colors[int(error)])
            # Plot the stabilizers between the data and syndrome qubits
            for (index, pauli) in qubit.acted_on_by:
                target = self._from_index(
                    self.data["synd_qubits"], index)
                bx, by, bz = target.location
                # Check that the syndrome qubit is not hidden
                if bx is None:
                    continue
                line_style = '-' if \
                    'IXYZ'[int(error)] != pauli.upper() and error != 0 \
                    else ':'
                if target.name is None:
                    axis.plot([ax, bx], [ay, by], [az, bz], line_style, lw=marker_size/10,
                            color=self.stabilizer_colors[pauli], zorder=0)
                else:
                    axis.plot([ax, bx], [ay, by], [az, bz], line_style, lw=marker_size/10,
                            color=self.stabilizer_colors[pauli], zorder=0, alpha=.1)
        # Plot the syndrome qubits
        syndromes = jax_get_syndromes(*self.transformation_matrix(), errors)
        for qubit, syndrome in zip(self._flatten(self.data["synd_qubits"]), syndromes):
            ax, ay, az = qubit.location
            if qubit.name is None:
                axis.plot(ax, ay, az, 's', ms=marker_size/2, c='black',
                          mfc='gray' if syndrome else 'black')
            else:
                axis.plot(ax, ay, az, 'h', ms=marker_size, c='black',
                          mfc='gray' if syndrome else 'black')
                axis.text(ax, ay, az, s=qubit.name, fontsize=marker_size/2,
                          ha='center', va='center', zorder=2,
                          color='black' if syndrome else 'gray')
        # Create legend
        axis.plot(0, 0, 0, ms=0,
                  color=self.stabilizer_colors['X'], label='X stabilizer')
        axis.plot(0, 0, 0, ms=0,
                  color=self.stabilizer_colors['Y'], label='Y stabilizer')
        axis.plot(0, 0, 0, ms=0,
                  color=self.stabilizer_colors['Z'], label='Z stabilizer')
        axis.legend(ncol=3)
        # Configure x-axis
        axis.set_xlabel('X')
        xmin, xmax = axis.get_xlim()
        axis.set_xticks(range(int(np.floor(xmin)), int(np.ceil(xmax)+1), 1))
        # Configure y-axis
        axis.set_ylabel('Y')
        ymin, ymax = axis.get_ylim()
        axis.set_yticks(range(int(np.floor(ymin)), int(np.ceil(ymax)+1), 1))
        # Configure z-axis
        axis.set_zlabel('Z')
        axis.set_zlim(0, 1)
        axis.set_zticks([0, 1])
        # Configure plot
        axis.view_init(elev, azim-90, roll)
        axis.set_aspect('equal')
        axis.set_title(title)
        # Return the axis for further potentially further processing
        return axis

    def qc_cycle(self) -> QuantumCircuit:
        """
        Creates the qiskit quantum circuit for the QEC code
        """

        def assign(qc_qubits: list, qec_qubits, i=0):
            if isinstance(qec_qubits, Qubit):
                # Ignore this qubit
                if qec_qubits.location[0] is None:
                    return i
                # Assign a qubit to this qubit
                qec_qubits.qubit = qc_qubits[i]
                return i+1
            # Handle list of qubits
            for qec_qubits_ in qec_qubits:
                i = assign(qc_qubits, qec_qubits_, i)
            return i

        data = QuantumRegister(size=self._size(
            self.data["data_qubits"]), name="DataQubits")
        meas = AncillaRegister(size=self._size(
            self.data["synd_qubits"]), name="MeasQubits")
        synd = ClassicalRegister(size=len(meas), name="Syndrome")
        assign(data, self.data["data_qubits"])
        assign(meas, self.data["synd_qubits"])
        circ = QuantumCircuit(data, meas, synd)
        circ.reset(meas)
        circ.h(meas)
        circ.barrier(meas)
        for target in self._flatten(self.data["data_qubits"]):
            for (index, pauli) in target.acted_on_by:
                control = self._from_index(self.data["synd_qubits"], index)
                if control.qubit is None:
                    continue
                match pauli:
                    case "X":
                        circ.cx(control.qubit, target.qubit)
                    case "Y":
                        circ.cy(control.qubit, target.qubit)
                    case "Z":
                        circ.cz(control.qubit, target.qubit)
        circ.barrier(meas)
        circ.h(meas)
        circ.measure(meas, synd)
        return circ


def surface_code_data(d: int) -> dict[str, list[list | Qubit]]:
    """
    Generates the qec data for a distance d XZ surface code
    """
    P = ['X', 'Z']  # For the XZ surface code
    return {
        "data_qubits": [[
            Qubit((i, j, 0), acted_on_by=[
                [[i, j], P[(i+j+1) % 2]],
                [[i+1, j], P[(i+j) % 2]],
                [[i, j+1], P[(i+j) % 2]],
                [[i+1, j+1], P[(i+j+1) % 2]]])
            for j in range(d)] for i in range(d)],
        "synd_qubits": [[
            Qubit((i-.5, j-.5, 0))
            if ((i+j+1) % 2 and (0 < i < d) or (i+j) % 2 and (0 < j < d))
            else Qubit((None, None, None))
            for j in range(d+1)] for i in range(d+1)],
    }


def surface_code_data_wls(d: int) -> dict[str, list[list | Qubit]]:
    """
    Generates the qec data for a distance d XZ surface code

    wls: with logical stabilizers (detects if a logical X or Z error has occured)
    """
    P = ['X', 'Z']  # For the XZ surface code
    return {
        "data_qubits": [[
            Qubit((i, j, 0), acted_on_by=[
                [[i, j], P[(i+j+1) % 2]],
                [[i+1, j], P[(i+j) % 2]],
                [[i, j+1], P[(i+j) % 2]],
                [[i+1, j+1], P[(i+j+1) % 2]]
            ] +
                ([[[d+1], 'Z']] if i == 0 else []) +
                ([[[d+2], 'X']] if j == 0 else [])
            )
            for j in range(d)] for i in range(d)],
        "synd_qubits": [[
            Qubit((i-.5, j-.5, 0))
            if ((i+j+1) % 2 and (0 < i < d) or (i+j) % 2 and (0 < j < d))
            else Qubit((None, None, None))
            for j in range(d+1)] for i in range(d+1)] + [
                Qubit((-1.5, -1.5, 0), name="X", connections=False),
                Qubit((-.5, -1.5, 0), name="Z", connections=False)
        ],
    }


def repetion_code_data(d: int, pauli: str):
    return {
        "data_qubits": [
            Qubit((i, 0, 0), acted_on_by=[
                [[i], pauli.upper()],
                [[i+1], pauli.upper()]])
            for i in range(d)],
        "synd_qubits": [
            Qubit((i-.5, 0, 0))
            if 0 < i < d
            else Qubit((None, None, None))
            for i in range(d+1)],
    }

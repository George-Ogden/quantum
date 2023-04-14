from ..gates import ControlledGate, Gate, Hadamard, Oracle, SWAP

from .qft import InverseQuantumFourierTransform as IQFT
from .circuit import Circuit

class QuantumPhaseEstimation(Circuit):
    """Quantum Phase Estimation (QPE) Circuit
    find the phase of an eigenvalue of a unitary operator
    for more information see: https://www.cl.cam.ac.uk/teaching/2223/QuantComp/Quantum_Computing_Lecture_9_2023.pdf
    """
    def __init__(self, t: int, controlled_gate: ControlledGate):
        """create a QPE circuit to estimate the phase of an eigenvalue of a unitary operator to t bits of precision

        Args:
            t (int): number of bits used in estimation
            controlled_gate (ControlledGate): a controlled unitary gate
        """
        # TODO: run with more controlled qubits
        assert controlled_gate.t == 1, "Controlled unity must have a single controlled qubit gate"
        # apply hadamard gate to all inputs
        gates = [Gate.parallel(*[Hadamard] * t) + Gate.Identity(controlled_gate.n)]
        # swap the qubits of the oracle to put the control qubit first
        oracle_swap = Gate.serial(
            *[
                Gate.Identity(i) + SWAP + Gate.Identity(controlled_gate.n - i - 1)
                for i in range(controlled_gate.n)
            ]
        )
        # modify the oracle
        oracle = oracle_swap * controlled_gate * oracle_swap.inverse
        for i in range(t):
            # move the qubit to the correct position
            swap = Gate.Identity(t - i - 1) + Gate.Swap(i) + Gate.Identity(controlled_gate.n)
            # apply the orracle 2^i times on the qubit
            gates.extend([
                swap,
                Gate.Identity(t - 1) + Gate.serial(*[oracle] * (2 ** i)),
                swap
            ])

        # apply inverse QFT to the qubits
        gates.append(IQFT(t).to_gate() + Gate.Identity(controlled_gate.n))
        super().__init__(gates, "Quantum Phase Estimation")
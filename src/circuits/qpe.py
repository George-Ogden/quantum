from ..gates import ControlledGate, Gate, Hadamard, SWAP
from ..oracle import Oracle

from .circuit import Circuit
from .qft import InverseQuantumFourierTransform as IQFT

class QuantumPhaseEstimation(Circuit):
    def __init__(self, t: int, controlled_gate: ControlledGate):
        assert controlled_gate.t == 1, "Controlled unity must have a single controlled qubit gate"
        gates = []
        gates.append(Gate.parallel(*[Hadamard] * t) + Gate.Identity(controlled_gate.n))
        big_swap = []
        for i in range(controlled_gate.n):
            big_swap.append(
                (Gate.Identity(i) + SWAP + Gate.Identity(controlled_gate.n - i - 1))
            )
        big_swap = Gate.serial(*big_swap)
        oracle = big_swap * controlled_gate * big_swap.inverse
        for i in range(t):
            swap = Gate.Identity(t - i - 1) + Gate.Swap(i) + Gate.Identity(controlled_gate.n)
            print(swap)
            gates.append(
                swap
            )
            gates.append(
                Gate.Identity(t - 1) + Gate.serial(*[oracle] * (2 ** i))
            )
            gates.append(
                swap
            )

        gates.append(IQFT(t).to_gate() + Gate.Identity(controlled_gate.n))
        super().__init__(gates, "Quantum Phase Estimation")
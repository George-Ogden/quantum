from ..gates import ControlledGate, Gate, Hadamard, SWAP
from ..oracle import Oracle

from .circuit import Circuit
from .qft import InverseQuantumFourierTransform as IQFT

class QuantumPhaseEstimation(Circuit):
    def __init__(self, t: int, controlled_gate: ControlledGate):
        assert controlled_gate.t == 1, "Controlled unity must have a single controlled qubit gate"
        gates = [Gate.parallel(*[Hadamard] * t) + Gate.Identity(controlled_gate.n)]
        oracle_swap = Gate.serial(
            *[
                Gate.Identity(i) + SWAP + Gate.Identity(controlled_gate.n - i - 1)
                for i in range(controlled_gate.n)
            ]
        )
        oracle = oracle_swap * controlled_gate * oracle_swap.inverse
        for i in range(t):
            swap = Gate.Identity(t - i - 1) + Gate.Swap(i) + Gate.Identity(controlled_gate.n)
            gates.extend([
                swap,
                Gate.Identity(t - 1) + Gate.serial(*[oracle] * (2 ** i)),
                swap
            ])

        gates.append(IQFT(t).to_gate() + Gate.Identity(controlled_gate.n))
        super().__init__(gates, "Quantum Phase Estimation")
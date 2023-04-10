from ..gate import Gate, Hadamard, Identity

from .circuit import Circuit

class QuantumFourierTransform(Circuit):
    def __init__(self, n: int):
        gates = []
        for i in range(n):
            gates.append(Gate.Identity(i) + Hadamard + Gate.Identity(n - i - 1))
            for j in range(i + 1, n):
                swap = Identity + (Gate.Swap(j - i - 1) if j > i + 1 else Gate.Identity(1))
                gates.append(Gate.Identity(i) + (swap * (Gate.CROT(j - i + 1) + Gate.Identity(j - i - 1)) * swap) + Gate.Identity(n - j - 1))
        for i in range(n // 2):
            gates.append(Gate.Identity(i) + Gate.Swap(n - 2 * i - 1) + Gate.Identity(i))
        super().__init__(gates, "QFT")
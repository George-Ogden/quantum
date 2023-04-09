from ..gate import Gate, Hadamard, Identity
from ..qubit import Qubit, Minus
from ..circuit import Circuit
from ..oracle import Oracle

from .algorithm import Algorithm

class DeutschJoszaAlgorithm(Algorithm):
    def __init__(self, oracle: Oracle):
        self.oracle = oracle
        self.n = oracle.n
        super().__init__("Deutsch-Jozsa")
    
    def build_circuit(self) -> Circuit:
        large_hadamard = Gate.parallel(*[Hadamard] * self.n)
        return Circuit([
            large_hadamard + Identity,
            Gate.Identity(self.n) + Hadamard,
            self.oracle
        ], "Deutsch-Jozsa")
    
    def get_start_state(self) -> Qubit:
        return Qubit.from_value(0, length=self.n, name="x") + Qubit.from_value(1, length=1, name="y")
    
    def measure(self, qubit: Qubit) -> float:
        return qubit.measure(basis=Minus, bit=self.n)
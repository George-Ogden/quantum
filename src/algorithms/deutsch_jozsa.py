import jax.numpy as jnp

from ..oracle import Oracle
from ..gate import Gate, Hadamard
from ..qubit import Qubit
from ..circuit import Circuit

from .algorithm import Algorithm

class DeutschJoszaAlgorithm(Algorithm):
    def __init__(self, oracle: Oracle):
        self.oracle = oracle
        self.n = oracle.n
        super().__init__("Deutsch-Jozsa")
    
    def build_circuit(self) -> Circuit:
        large_hadamard = Gate.parallel(*[Hadamard] * self.n)
        return Circuit([
            large_hadamard + Gate.Identity(1),
            Gate.Identity(self.n) + Hadamard,
            self.oracle
        ], "Deutsch-Jozsa")
    
    def get_start_state(self) -> Qubit:
        return Qubit.from_value(0, length=self.n, name="x") + Qubit.from_value(1, length=1, name="y")
    
    def measure(self, qubit: Qubit) -> float:
        return qubit.measure(basis=jnp.array([1, -1]) / jnp.sqrt(2), bit=self.n)
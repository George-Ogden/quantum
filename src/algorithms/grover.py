import jax.numpy as jnp

from typing import List

from ..utils import hermitian, to_matrix
from ..qubit import Qubit, Minus, Plus, One
from ..gates import Gate, Hadamard, Identity
from ..circuits.circuit import Circuit
from ..oracle import Oracle

from .algorithm import Algorithm

class GroversAlgorithm(Algorithm):
    def __init__(self, oracle: Oracle):
        self.oracle = oracle
        self.n = oracle.n
        super().__init__("Grover's")
    
    def build_circuit(self) -> Circuit:
        large_hadamard = Gate.parallel(*[Hadamard] * self.n)
        large_plus = Qubit.entangle(*[Plus] * self.n).vector
        W = Gate(
                matrix=2 * jnp.outer(to_matrix(large_plus), hermitian(large_plus))
                  - jnp.identity(2 ** self.n),
                name="W"
        )
        iterations = int(jnp.ceil(jnp.pi * jnp.sqrt(2 ** self.n) / 4))
        return Circuit([
            large_hadamard + Identity,
            *[self.oracle * (W + Identity)] * iterations
        ], "Grover")
    
    def get_start_state(self) -> Qubit:
        return Qubit.from_value(0, length=self.n) + Minus
    
    def measure(self, qubit: Qubit) -> List[float]:
        return [
            qubit.measure(basis=One, bit=i)
            for i in reversed(range(1, self.n + 1))
        ]
import jax.numpy as jnp

from typing import List

from ..gates import Gate, Hadamard, Identity, Oracle
from ..qubit import Qubit, Minus, Plus, One
from ..utils import hermitian, to_matrix
from ..circuits.circuit import Circuit

from .algorithm import Algorithm

class GroversAlgorithm(Algorithm):
    """Grover's Algorithm
    find the unique significant value with a quadratic speedup over classical algorithms
    for more information see: https://www.cl.cam.ac.uk/teaching/2223/QuantComp/Quantum_Computing_Lecture_8_2023.pdf
    """
    def __init__(self, oracle: Oracle):
        self.oracle = oracle
        self.n = oracle.n
        super().__init__("Grover's")
    
    def build_circuit(self) -> Circuit:
        # create a large hadamard gate to apply to the input qubits
        large_hadamard = Gate.parallel(*[Hadamard] * self.n)
        # create a large plus to use as a to generate the W gate
        large_plus = Qubit.entangle(*[Plus] * self.n).vector
        # create a W gate to increase reflection
        W = Gate(
            matrix=2 * jnp.outer(to_matrix(large_plus), hermitian(large_plus))
            - jnp.identity(2 ** self.n),
            name="W"
        )
        # calculate the number of iterations to perform
        iterations = int(jnp.ceil(jnp.pi * jnp.sqrt(2 ** self.n) / 4))
        return Circuit([
            # create superposition of inputs
            large_hadamard + Identity,
            # apply oracle and W gate multiple times
            *[self.oracle * (W + Identity)] * iterations
        ], "Grover")
    
    def get_start_state(self) -> Qubit:
        # start with all inputs set to 0 and the control qubit in the minus superposition
        return Qubit.from_value(0, length=self.n) + Minus
    
    def measure(self, qubit: Qubit) -> List[float]:
        return [
            # measure the input qubits in the One basis
            qubit.measure(basis=One, bit=i)
            for i in reversed(range(1, self.n + 1))
        ]
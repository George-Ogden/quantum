from ..gates import Gate, Hadamard, Oracle
from ..circuits.circuit import Circuit
from ..qubit import Qubit, Minus

from .algorithm import Algorithm

class DeutschJoszaAlgorithm(Algorithm):
    """Deutsch-Jozsa algorithm
    determine whether a function is balanced or constant with an exponential speedup over classical algorithms
    for more information see: https://www.cl.cam.ac.uk/teaching/2223/QuantComp/Quantum_Computing_Lecture_7_2023.pdf
    """
    def __init__(self, oracle: Oracle):
        self.oracle = oracle
        self.n = oracle.n
        super().__init__("Deutsch-Jozsa")
    
    def build_circuit(self) -> Circuit:
        return Circuit([
            # apply hadamard gate to all inputs
            Gate.parallel(*[Hadamard] * (self.n + 1)),
            # pass through oracle
            self.oracle
        ], "Deutsch-Jozsa")
    
    def get_start_state(self) -> Qubit:
        # start with all inputs set to 0 and the control qubit set to 1
        return Qubit.from_value(0, length=self.n, name="x") + Qubit.from_value(1, length=1, name="y")
    
    def measure(self, qubit: Qubit) -> float:
        # measure the control qubit in the Minus basis
        return qubit.measure(basis=Minus, bit=self.n)
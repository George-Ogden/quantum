from src.circuit import Circuit
from src.qubit import Qubit

from abc import ABC, abstractmethod

class Algorithm(metaclass=ABC):
    def __init__(self, circuit: Circuit):
        self.circuit = circuit
    
    @abstractmethod
    def run(self, start_state: Qubit) -> Qubit:
        return self.circuit(start_state)
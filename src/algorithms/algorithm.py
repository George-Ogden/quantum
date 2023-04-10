from src.circuits.circuit import Circuit
from src.qubit import Qubit

from abc import ABC, abstractmethod
from typing import Optional

class Algorithm(ABC):
    def __init__(self, name: Optional[str] = None):
        self.circuit = self.build_circuit()
        self.start_state = self.get_start_state()
        self.name = (name or self.__class__.__name__).title()

    @abstractmethod
    def build_circuit(self) -> Circuit:
        ...

    @abstractmethod
    def get_start_state(self) -> Qubit:
        ...

    def __call__(self, *args, **kwargs) -> Qubit:
        return self.run(*args, **kwargs)

    @abstractmethod
    def measure(self, qubit: Qubit) -> float:
        ...

    def run(self) -> Qubit:
        return self.measure(self.circuit(self.start_state))

    def __repr__(self) -> str:
        return f"{self.name} Algorithm ({self.circuit})"
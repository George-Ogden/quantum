from abc import ABC, abstractmethod
from typing import Optional, List

from ..circuits.circuit import Circuit
from ..qubit import Qubit

class Algorithm(ABC):
    """Abstract base class for algorithms"""
    def __init__(self, name: Optional[str] = None):
        # initialises by building circuits and getting start states
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
        """Runs the algorithm"""
        return self.run(*args, **kwargs)

    @abstractmethod
    def measure(self, qubit: Qubit) -> List[float]:
        ...

    def run(self) -> Qubit:
        """Runs the algorithm"""
        return self.measure(self.circuit(self.start_state))

    def __repr__(self) -> str:
        return f"{self.name} Algorithm ({self.circuit})"
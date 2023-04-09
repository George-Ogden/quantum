from __future__ import annotations

import jax.numpy as jnp

from typing import Optional, Union

from src.qubit import Qubit
from src.gate import Gate

class Circuit:
    def __init__(self, gates: list[Gate], name: Optional[str] = None):
        self.gates = gates
        self.name = (name or "Circuit").title()
    
    def __call__(self, qubit: Qubit) -> Qubit:
        for gate in self.gates:
            qubit = gate(qubit)
            print(qubit)
        return qubit
    
    def __mul__(self, other: Union[Circuit, Gate, jnp.ndarray]) -> Circuit:
        """Puts the gate in series with another circuit or gate"""
        if isinstance(other, jnp.ndarray):
            other = Gate(other)
        if isinstance(other, Circuit):
            return Circuit(self.gates + other.gates, name=f"{self.name} * {other.name}")
        elif isinstance(other, Gate):
            return Circuit(self.gates + [other], name=f"{self.name}")
        raise TypeError(f"Cannot multiply Circuit with {type(other)}")
    
    def __add__(self, other: Union[Circuit, Gate, jnp.ndarray]) -> Gate:
        """Puts the circuit in parallel with another circuit or gate"""
        if isinstance(other, jnp.ndarray):
            other = Gate(other)
        if isinstance(other, Gate):
            other = Circuit([other], other.name)
        if isinstance(other, Circuit):
            return Circuit([Gate.parallel(self.to_gate(), other.to_gate())], name=f"{self.name} + {other.name}")
        raise TypeError(f"Cannot add Circuit with {type(other)}")

    def __matmul__(self, qubit: Qubit) -> Qubit:
        """Applies the circuit to the given qubit"""
        return self(qubit)

    def __eq__(self, other: Union[Circuit, Gate, jnp.ndarray]) -> bool:
        """Checks if the circuit is equal to another circuit or gate"""
        if isinstance(other, Circuit):
            other = other.to_gate()
        elif isinstance(other, jnp.ndarray):
            other = Gate(other)
        if not isinstance(other, Gate):
            return False
        return self.to_gate() == other

    def __len__(self) -> int:
        """Returns the number of gates in the circuit"""
        return len(self.gates)

    def __repr__(self) -> str:
        """Returns a string representation of the circuit"""
        return f"{self.name} {self.gates}"
    
    def to_gate(self) -> Gate:
        """Converts the circuit to a gate"""
        return Gate.serial(*self.gates)
from __future__ import annotations

import jax.numpy as jnp

from .qubit import Qubit

from typing import Optional, Union

class Gate:
    def __init__(self, matrix: jnp.ndarray, name: Optional[str] = None):
        self.name = (name or "gate").upper()
        self.matrix = matrix

    def __repr__(self) -> str:
        """Returns a string representation of the gate"""
        return f"{self.name} {self.matrix.shape}"

    def __call__(self, x: Qubit) -> Qubit:
        """Applies the gate to the given qubit"""
        return Qubit(self.matrix @ x.vector, name=x.name)

    def __mul__(self, other: Union[Gate, jnp.ndarray]) -> Gate:
        """Puts the gate in series with another gate"""
        if isinstance(other, jnp.ndarray):
            other = Gate(other)
        if isinstance(other, Gate):
            return Gate(self.matrix @ other.matrix, name=f"{self.name} * {other.name}")
        raise TypeError(f"Cannot multiply Gate with {type(other)}")
    
    def __add__(self, other: Union[Gate, jnp.ndarray]) -> Gate:
        """Puts the gate in parallel with another gate"""
        if isinstance(other, jnp.ndarray):
            other = Gate(other)
        if isinstance(other, Gate):
            return Gate(jnp.kron(self.matrix, other.matrix), name=f"{self.name} + {other.name}")
        raise TypeError(f"Cannot add Gate with {type(other)}")

    def __matmul__(self, qubit: Qubit) -> Qubit:
        """Applies the gate to the given qubit"""
        return self(qubit)

    def __eq__(self, gate: Union[Gate, jnp.ndarray]) -> bool:
        """Checks if the gate is equal to another gate"""
        if isinstance(gate, jnp.ndarray):
            gate = Gate(gate)
        if not isinstance(gate, Gate):
            return False
        return self.matrix.shape == gate.matrix.shape and jnp.allclose(self.matrix, gate.matrix)

    @property
    def n(self) -> int:
        """Returns the number of qubits the gate acts on"""
        return int(jnp.ceil(jnp.log2(self.matrix.shape[0])))

    @staticmethod
    def Identity(n: int) -> Gate:
        return Gate(jnp.eye(2 ** n), "identity")
    
    @staticmethod
    def parallel(*gates: Gate) -> Gate:
        """Puts the gates in parallel"""
        sum = gates[0]
        for gate in gates[1:]:
            sum = sum + gate
        return sum

    @staticmethod
    def serial(*gates: Gate) -> Gate:
        """Puts the gates in series"""
        product = gates[0]
        for gate in gates[1:]:
            product = product * gate
        return product

Hadamard = H = Gate(jnp.array([[1, 1], [1, -1]]), "H")
Identity = I = Gate.Identity(1)
Pauli_X = X = Gate(jnp.array([[0, 1], [1, 0]]), "X")
CNOT = Gate(jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), "CNOT")
Neutral = Gate(jnp.array([1]), "neutral")
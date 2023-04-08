from __future__ import annotations

import jax.numpy as jnp

from .qubit import Qubit

from typing import Optional, Union

class Gate:
    def __init__(self, matrix: jnp.ndarray, name: Optional[str] = None):
        self.name = (name or "gate").upper()
        self.matrix = matrix

    def __repr__(self):
        """Returns a string representation of the gate"""
        return f"{self.name} {self.matrix.shape}"

    def __call__(self, x: Qubit):
        """Applies the gate to the given state"""
        return Qubit(self.matrix @ x.vector, name=x.name)

    def __mul__(self, other: Union[Gate, jnp.ndarray]) -> Union[Gate, jnp.ndarray]:
        """Multiplies the gate with another gate or a state"""
        if isinstance(other, Gate):
            return Gate(self.matrix @ other.matrix, name=f"{self.name} * {other.name}")
        elif isinstance(other, Qubit):
            return self(other)
        raise TypeError(f"Cannot multiply Gate with {type(other)}")
    
    def __add__(self, other: Union[Gate, jnp.ndarray]) -> Gate:
        """Puts the gate in parallel with another gate"""
        if isinstance(other, Gate):
            return Gate(jnp.kron(self.matrix, other.matrix), name=f"{self.name} + {other.name}")
        elif isinstance(other, jnp.ndarray):
            return self + Gate(other)
        raise TypeError(f"Cannot add Gate with {type(other)}")

    def __matmul__(self, qubit: Qubit) -> jnp.ndarray:
        """Applies the gate to the given state"""
        return self(qubit)

    @staticmethod
    def Identity(n: int) -> Gate:
        return Gate(jnp.eye(2 ** n), "identity")

hadamard = H = Gate(jnp.array([[1, 1], [1, -1]]), "H")
identity = I = Gate.Identity(1)
pauli_x = X = Gate(jnp.array([[0, 1], [1, 0]]), "X")
CNOT = Gate(jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), "CNOT")
neutral = Gate(jnp.array([1]), "neutral")
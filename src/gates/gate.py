from __future__ import annotations

import jax.numpy as jnp

from typing import Optional, Union

from ..qubit import Qubit

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
            return Gate(self.matrix @ other.matrix, name=f"({self.name} * {other.name})")
        raise TypeError(f"Cannot multiply Gate with {type(other)}")
    
    def __add__(self, other: Union[Gate, jnp.ndarray]) -> Gate:
        """Puts the gate in parallel with another gate"""
        if isinstance(other, jnp.ndarray):
            other = Gate(other)
        if isinstance(other, Gate):
            return Gate(jnp.kron(self.matrix, other.matrix), name=f"({self.name} + {other.name})")
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
    def inverse(self) -> Gate:
        """Returns the inverse of the gate"""
        return Gate(jnp.linalg.inv(self.matrix), f"inv({self.name})")

    @property
    def n(self) -> int:
        """Returns the number of qubits the gate acts on"""
        return int(jnp.ceil(jnp.log2(self.matrix.shape[0])))

    @staticmethod
    def Identity(n: int) -> Gate:
        return Gate(jnp.eye(int(2 ** n)), "identity")
    
    @staticmethod
    def Swap(n: int) -> Gate:
        """Returns a gate that swaps the 0th and nth qubits"""
        if n == 0:
            return Gate.Identity(1)
        swap = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        matrix = jnp.eye(2 ** (n + 1))
        for i in range(n - 1):
            matrix @= jnp.kron(jnp.eye(2 ** i), jnp.kron(swap, jnp.eye(2 ** (n - i - 1))))
        return Gate(matrix @ jnp.kron(jnp.eye(2 ** (n - 1)), swap) @ jnp.linalg.inv(matrix), "SWAP")

    @staticmethod
    def R(n: int) -> Gate:
        return Gate(jnp.array([[1, 0], [0, jnp.exp(2 * jnp.pi * 1j / (2 ** n))]]), f"R({n})")

    @staticmethod
    def CROT(n: int) -> Gate:
        """Returns a gate that controls the given gate"""
        return Gate(jnp.block([[Gate.Identity(1).matrix, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), Gate.R(n).matrix]]), f"CROT({n})")
    
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
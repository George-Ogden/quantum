from __future__ import annotations

import jax.numpy as jnp

from typing import Optional, Union

class Gate:
    def __init__(self, matrix: jnp.ndarray, name: Optional[str] = None):
        self.name = (name or "gate").title()
        self.matrix = matrix

    def __repr__(self):
        return f"{self.name} {self.matrix.shape}"

    def __call__(self, x: jnp.ndarray):
        y = (self.matrix @ x)
        return y / jnp.linalg.norm(y) * jnp.linalg.norm(x)

    def __mul__(self, other: Union[Gate, jnp.ndarray]) -> Union[Gate, jnp.ndarray]:
        if isinstance(other, Gate):
            return Gate(self.matrix @ other.matrix, name=f"{self.name} * {other.name}")
        elif isinstance(other, jnp.ndarray):
            return self(other)
        raise TypeError(f"Cannot multiply Gate with {type(other)}")
    
    def __add__(self, other: Union[Gate, jnp.ndarray]) -> Gate:
        if isinstance(other, Gate):
            return Gate(jnp.kron(self.matrix, other.matrix), name=f"{self.name} + {other.name}")
        elif isinstance(other, jnp.ndarray):
            return self + Gate(other)
        raise TypeError(f"Cannot add Gate with {type(other)}")

    def __matmul__(self, qubit: jnp.ndarray) -> jnp.ndarray:
        return self(qubit)

    @staticmethod
    def Identity(n: int) -> Gate:
        return Gate(jnp.eye(2 ** n), "identity")

class Oracle(Gate):
    @staticmethod
    def from_table(table: jnp.ndarray) -> Oracle:
        n = int(jnp.log2(len(table)))
        U = jnp.zeros((2**(n + 1), 2**(n + 1)))
        for x in range(2**n):
            y = table[x]
            U = U.at[x * 2, x * 2].set(1 - y)
            U = U.at[x * 2 + 1, x * 2 + 1].set(1 - y)
            U = U.at[x * 2, x * 2 + 1].set(y)
            U = U.at[x * 2 + 1, x * 2].set(y)
        return Oracle(U, "oracle")


hadamard = Gate(jnp.array([[1, 1], [1, -1]]), "hadamard")
identity = Gate.Identity(1)
pauli_x = Gate(jnp.array([[0, 1], [1, 0]]), "pauli_x")
CNOT = Gate(jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), "CNOT")
neutral = Gate(jnp.array([1]), "neutral")
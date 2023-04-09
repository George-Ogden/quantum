from __future__ import annotations
import jax.numpy as jnp

from typing import Optional, Union

from .utils import hermitian, to_matrix

class Qubit:
    def __init__(self, vector: jnp.ndarray, name: Optional[str] = None):
        self.name = (name or "qubit").lower()
        self.vector = vector / jnp.linalg.norm(vector)
    
    @staticmethod
    def from_value(value: int, length: int = 1, name: Optional[str] = None) -> Qubit:
        vector = jnp.zeros(2 ** length, dtype=jnp.float32)
        vector = vector.at[value].set(1)
        return Qubit(vector, name)
    
    def __repr__(self):
        """Returns a string representation of the qubit"""
        return f"{self.name} {self.vector.tolist()}"
    
    def __add__(self, other: Qubit) -> Qubit:
        """Puts the qubit in an entangled state with another qubit"""
        return Qubit(jnp.kron(self.vector, other.vector), name=f"{self.name} + {other.name}")

    def __eq__(self, qubit: Union[Qubit, jnp.ndarray]) -> bool:
        """Checks if the qubit is equal to another qubit"""
        if isinstance(qubit, jnp.ndarray):
            qubit = Qubit(qubit)
        if not isinstance(qubit, Qubit):
            return False
        return len(self.vector) == len(qubit.vector) and jnp.allclose(self.vector, qubit.vector)

    @property
    def n(self) -> int:
        """Returns the number of qubits the gate acts on"""
        return int(jnp.ceil(jnp.log2(self.matrix.shape[0])))

    def measure(self, basis: Optional[Union[Qubit, jnp.ndarray, int]] = None, bit: int = 0):
        """Measures the qubit"""
        # default to 1 as the basis
        if basis is None:
            basis = Qubit.from_value(1)
        elif isinstance(basis, int):
            basis = Qubit.from_value(basis)

        # extract the vector from the basis
        if isinstance(basis, Qubit):
            basis = basis.vector
        
        # convert basis to matrix
        if basis.ndim == 1:
            # convert bit to selection of bits
            new_basis = jnp.zeros((len(self.vector), len(self.vector)), dtype=jnp.float32)
            for i in range(len(self.vector)):
                j = i | (1 << bit)
                if i == j:
                    continue
                base = jnp.zeros_like(self.vector)
                base = base.at[i].set(basis[0])
                base = base.at[j].set(basis[1])
                new_basis += jnp.outer(to_matrix(base), hermitian(base))

            basis = new_basis

        if len(basis) == 2:
            n = jnp.log2(len(self.vector)).astype(int)
            basis = jnp.kron(jnp.eye(n - bit), jnp.kron(basis, jnp.eye(bit + 1)))

        return (hermitian(self.vector) @ hermitian(basis) @ basis @ to_matrix(self.vector)).reshape(())
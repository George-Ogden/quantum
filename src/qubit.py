from __future__ import annotations
import jax.numpy as jnp

from typing import Optional, Union

from .utils import hermitian, to_matrix

class Qubit:
    def __init__(self, vector: jnp.ndarray, name: Optional[str] = None):
        self.name = (name or "qubit").lower()
        self.vector = vector / jnp.linalg.norm(vector)
    
    @staticmethod
    def from_value(value: int, length: int, name: Optional[str] = None) -> Qubit:
        vector = jnp.zeros(2 ** length, dtype=jnp.float32)
        vector = vector.at[value].set(1)
        return Qubit(vector, name)
    
    def __repr__(self):
        """Returns a string representation of the qubit"""
        return f"{self.name} {self.vector.tolist()}"
    
    def __add__(self, other: Qubit) -> Qubit:
        """Puts the qubit in an entangled state with another qubit"""
        return Qubit(jnp.kron(self.vector, other.vector), name=f"{self.name} + {other.name}")

    def measure(self, bit: int = 0, basis: Optional[Union[Qubit, jnp.ndarray]] = None):
        """Measures the qubit"""
        # default to 1 as the basis
        if basis is None:
            basis = Qubit.from_value(1)

        # extract the vector from the basis
        if isinstance(basis, Qubit):
            basis = basis.vector
        
        # convert basis to matrix
        if basis.ndim == 1:
            # convert bit to selection of bits
            indices = jnp.arange(len(self.vector))
            basis = jnp.stack((indices & (1 << bit), indices & (1 << bit) ^ 1), axis=-1) * basis
            basis = basis.sum(axis=-1)
            basis = jnp.outer(to_matrix(basis), hermitian(basis))

        # TODO: update state after measurement
        return hermitian(self.vector) @ hermitian(basis) @ basis @ to_matrix(self.vector)
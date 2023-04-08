from __future__ import annotations
import jax.numpy as jnp

from typing import Optional

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
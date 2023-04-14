from __future__ import annotations
import jax.numpy as jnp

from typing import Optional, Union

from .utils import hermitian, to_matrix

class Qubit:
    """Class for representing a qubit"""
    def __init__(self, vector: jnp.ndarray, name: Optional[str] = None):
        """initialise a qubit from a vector

        Args:
            vector (jnp.ndarray): state of the qubit
            name (Optional[str], optional): name of the qubit. Defaults to None.
        """
        self.name = (name or "qubit").lower()
        self.vector = vector / jnp.linalg.norm(vector)
    
    @staticmethod
    def from_value(value: int, length: int = 1, name: Optional[str] = None) -> Qubit:
        """initialise a qubit from a value

        Args:
            value (int): value of the qubit
            length (int, optional): length of the qubit to return. Defaults to 1.
            name (Optional[str], optional): name of the qubit. Defaults to None.

        Returns:
            Qubit: a qubit whose value is the binary representation of value
        """
        # create a vector of zeros
        vector = jnp.zeros(2 ** length, dtype=jnp.float32)
        # set the value of the qubit
        vector = vector.at[value].set(1)
        return Qubit(vector, name)
    
    def __repr__(self):
        """Returns a string representation of the qubit"""
        return f"{self.name} {self.vector}"
    
    def __add__(self, other: Qubit) -> Qubit:
        """Puts the qubit in an entangled state with another qubit"""
        return Qubit(jnp.kron(self.vector, other.vector), name=f"{self.name} + {other.name}")

    def __eq__(self, qubit: Union[Qubit, jnp.ndarray]) -> bool:
        """Checks if the qubit is equal to another qubit"""
        if isinstance(qubit, jnp.ndarray):
            qubit = Qubit(qubit)
        if not isinstance(qubit, Qubit):
            return False
        return len(self.vector) == len(qubit.vector) and jnp.allclose(self.vector, qubit.vector, atol=1e-5)

    @property
    def n(self) -> int:
        """Returns the number of classical bits needed to represent the qubit"""
        return int(jnp.ceil(jnp.log2(self.matrix.shape[0])))

    def measure(self, basis: Optional[Union[Qubit, jnp.ndarray, int]] = None, bit: int = 0) -> float:
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

    @staticmethod
    def entangle(*qubits: Qubit) -> Qubit:
        """entangles a list of qubits"""
        return sum(qubits[1:], start=qubits[0])

# useful qubits
Zero = Qubit.from_value(0, name="zero")
One = Qubit.from_value(1, name="one")
Plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2), name="plus")
Minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2), name="minus")
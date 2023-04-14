from __future__ import annotations
import jax.numpy as jnp

from typing import Optional

from .gate import Gate

class Oracle(Gate):
    """A class representing an oracle gate"""
    def __init__(self, matrix: jnp.ndarray, name: Optional[str] = None):
        """create an oracle from a matrix

        Args:
            matrix (jnp.ndarray): matrix representing the oracle
            name (Optional[str], optional): name of the gate. Defaults to None.
        """
        super().__init__(matrix, name)
        self.name = (name or "Oracle").title()

    @staticmethod
    def from_table(table: jnp.ndarray) -> Oracle:
        """create an oracle from a table of values

        Args:
            table (jnp.ndarray): a jax array of values where table[i] == f(i)

        Returns:
            Oracle: an oracle that returns (x kron f(x) ^ y)
        """
        # calculate the number of qubits
        n = int(jnp.log2(len(table)))
        # create an oracle to operate on n + 1 qubits
        U = jnp.zeros((2**(n + 1), 2**(n + 1)))
        for x in range(2**n):
            # look up the value of f(x)
            y = table[x]
            # set the value of the oracle
            U = U.at[x * 2, x * 2].set(1 - y)
            U = U.at[x * 2 + 1, x * 2 + 1].set(1 - y)
            U = U.at[x * 2, x * 2 + 1].set(y)
            U = U.at[x * 2 + 1, x * 2].set(y)
        return Oracle(U)

    @property
    def n(self) -> int:
        # return the number of qubits the oracle operates on
        return super().n - 1
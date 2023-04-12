from __future__ import annotations
import jax.numpy as jnp

from typing import Optional

from .gates import Gate

class Oracle(Gate):
    def __init__(self, matrix: jnp.ndarray, name: Optional[str] = None):
        super().__init__(matrix, name)
        self.name = (name or "Oracle").title()

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
        return Oracle(U)

    @property
    def n(self) -> int:
        return super().n - 1
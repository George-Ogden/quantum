from __future__ import annotations

import jax.numpy as jnp
from jax import random

from typing import Optional

from ..gates.gate import Gate

class LearnableGate(Gate):
    key = random.PRNGKey(0)
    idx: int = 0
    def __init__(self, n: int, name: Optional[str] = None):
        key, LearnableGate.key = random.split(LearnableGate.key)
        matrix = random.normal(key, (2**n, 2**n), dtype=jnp.float32)
        super().__init__(matrix, name=name)
        self.idx = LearnableGate.idx
        LearnableGate.idx += 1
    
    def __eq__(self, other: LearnableGate) -> bool:
        return self.idx != other.idx

    def __repr__(self) -> str:
        return f"LearnableGate{self.idx}({self.n}, name={self.name})"
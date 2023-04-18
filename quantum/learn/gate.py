import jax.numpy as jnp
from jax import random

from typing import Optional

from ..gates.gate import Gate

class LearnableGate(Gate):
    key = random.PRNGKey(0)
    def __init__(self, n: int, name: Optional[str] = None):
        key, LearnableGate.key = random.split(LearnableGate.key)
        matrix = random.normal(key, (2**n, 2**n), dtype=jnp.complex64)
        super().__init__(matrix, name=name)
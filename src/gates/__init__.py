import jax.numpy as jnp

from .control import ControlledGate
from .gate import Gate

Hadamard = H = Gate(jnp.array([[1, 1], [1, -1]]), "H")
Identity = I = Gate.Identity(1)
Pauli_X = X = Gate(jnp.array([[0, 1], [1, 0]]), "X")
CNOT = Gate(jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), "CNOT")
SWAP = Gate.Swap(1)
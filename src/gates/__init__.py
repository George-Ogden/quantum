import jax.numpy as jnp

from .control import ControlledGate
from .gate import Gate

Hadamard = H = Gate(jnp.array([[1, 1], [1, -1]]), "H")
Identity = I = Gate.Identity(1)
Pauli_X = X = Gate(jnp.array([[0, 1], [1, 0]]), "X")
Pauli_Y = Y = Gate(jnp.array([[0, -1j], [1j, 0]]), "Y")
Pauli_Z = Z = Gate(jnp.array([[1, 0], [0, -1]]), "Z")
Phase = S = P = Gate(jnp.array([[1, 0], [0, 1j]]), "S")
T = Gate(jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]]), "T")
CNOT = Gate(jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), "CNOT")
SWAP = Gate.Swap(1)
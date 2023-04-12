import jax.numpy as jnp

from src.gates import ControlledGate, Gate, Pauli_X, CNOT, SWAP
from src.qubit import Qubit

def test_controlled_pauli_x_gate():
    gate = ControlledGate.from_gate(Pauli_X)
    assert gate == SWAP * CNOT * SWAP

def test_controlled_gate():
    unitary = Gate(jnp.array([[0, -1j], [1j, 0]]))
    gate = ControlledGate.from_gate(unitary)
    for i in range(2):
        for j in range(2):
            control = Qubit.from_value(i)
            qubit = Qubit.from_value(j)
            if i == 0:
                assert gate(qubit + control) == qubit + control
            else:
                assert gate(qubit + control) == (unitary @ qubit) + control
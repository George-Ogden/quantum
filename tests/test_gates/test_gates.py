import jax.numpy as jnp

from quantum.gates import *
from quantum.qubit import Qubit, One, Zero, Plus, Minus

def test_single_bit_Identity():
    gate = Gate.Identity(1)
    assert gate == jnp.array([[1, 0], [0, 1]])

def test_two_bit_Identity():
    gate = Gate.Identity(2)
    assert gate == jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def test_three_bit_Identity():
    gate = Gate.Identity(3)
    assert gate == jnp.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ])

def test_pauli_x_gate():
    gate = Pauli_X
    assert gate(Zero) == One
    assert gate(One) == Zero

def test_hadamard_gate():
    gate = Hadamard
    assert gate(Zero) == Plus
    assert gate(One) == Minus

def test_identity_gate():
    gate = Identity
    assert gate(Zero) == Zero
    assert gate(One) == One

def test_cnot_gate_0_1_basis():
    gate = CNOT
    assert gate(Zero + Zero) == Zero + Zero
    assert gate(Zero + One) == Zero + One
    assert gate(One + Zero) == One + One
    assert gate(One + One) == One + Zero

def test_cnot_gate_plus_minus_basis():
    gate = CNOT
    assert gate(Plus + Plus) == Plus + Plus
    assert gate(Plus + Minus) == Minus + Minus
    assert gate(Minus + Plus) == Minus + Plus
    assert gate(Minus + Minus) == Plus + Minus

def test_swap_gate():
    gate = SWAP
    qubit_0 = Qubit.from_value(0, length=2)
    qubit_1 = Qubit.from_value(1, length=2)
    qubit_2 = Qubit.from_value(2, length=2)
    qubit_3 = Qubit.from_value(3, length=2)

    assert gate(qubit_0) == qubit_0
    assert gate(qubit_1) == qubit_2
    assert gate(qubit_2) == qubit_1
    assert gate(qubit_3) == qubit_3
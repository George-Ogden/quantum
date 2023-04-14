import jax.numpy as jnp

from quantum.gates import *
from quantum.qubit import Qubit, One, Zero, Plus, Minus

def test_gate_creation_no_name():
    mat = jnp.array([[1, 0], [0, 1]])
    gate = Gate(mat)
    assert jnp.allclose(gate.matrix, mat)

def test_gate_creation_with_name():
    mat = jnp.array([[1, 0], [0, 1]])
    gate = Gate(mat, "I")
    assert gate.name == "I"
    assert jnp.allclose(gate.matrix, mat)

def test_gate_call():
    mat = jnp.array([[0, 1], [1, 0]])
    qubit = Qubit(jnp.array([1, 0]), "q")
    gate = Gate(mat)
    assert gate(qubit) == jnp.array([0, 1])

def test_gate_gate_mat_multiplication():
    gate_1 = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    gate_2 = Gate(jnp.array([[1, 0], [0, -1]]), "Z")
    assert gate_1 * gate_2 == jnp.array([[0, -1], [1, 0]])

def test_gate_qubit_mat_multiplication():
    qubit = Qubit(jnp.array([1, 0]), "q")
    gate = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    assert gate @ qubit == jnp.array([0, 1])

def test_gate_multiplication():
    gate_1 = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    gate_2 = Gate(jnp.array([[1, 0], [0, -1]]), "Z")
    assert gate_1 * gate_2 == jnp.array([[0, -1], [1, 0]])

def test_gate_gate_addition():
    gate_1 = Gate(jnp.array([[1, 0], [0, 1]]), "I")
    gate_2 = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    assert gate_1 + gate_2 == jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

def test_gate_repr(capsys):
    gate = Gate(jnp.array([[1, 0], [0, 1]]), "gate_name")
    print(gate)
    captured = capsys.readouterr()
    assert 'gate_name' in captured.out.lower()

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

def test_swap_gate_0():
    gate = Gate.Swap(0)
    assert gate(Zero) == Zero
    assert gate(One) == One
    assert gate(Plus) == Plus
    assert gate(Minus) == Minus

def test_swap_gate_2():
    gate = Gate.Swap(2)
    qubit_0 = Qubit.from_value(0, length=3)
    qubit_1 = Qubit.from_value(1, length=3)
    qubit_2 = Qubit.from_value(2, length=3)
    qubit_3 = Qubit.from_value(3, length=3)
    qubit_4 = Qubit.from_value(4, length=3)
    qubit_5 = Qubit.from_value(5, length=3)
    qubit_6 = Qubit.from_value(6, length=3)
    qubit_7 = Qubit.from_value(7, length=3)

    assert gate(qubit_0) == qubit_0
    assert gate(qubit_1) == qubit_4
    assert gate(qubit_2) == qubit_2
    assert gate(qubit_3) == qubit_6
    assert gate(qubit_4) == qubit_1
    assert gate(qubit_5) == qubit_5
    assert gate(qubit_6) == qubit_3
    assert gate(qubit_7) == qubit_7

def test_swap_gate_3():
    gate = Gate.Swap(3)
    qubit_0 = Qubit.from_value(0, length=4)
    qubit_1 = Qubit.from_value(1, length=4)
    qubit_2 = Qubit.from_value(2, length=4)
    qubit_3 = Qubit.from_value(3, length=4)
    qubit_4 = Qubit.from_value(4, length=4)
    qubit_5 = Qubit.from_value(5, length=4)
    qubit_6 = Qubit.from_value(6, length=4)
    qubit_7 = Qubit.from_value(7, length=4)
    qubit_8 = Qubit.from_value(8, length=4)
    qubit_9 = Qubit.from_value(9, length=4)
    qubit_10 = Qubit.from_value(10, length=4)
    qubit_11 = Qubit.from_value(11, length=4)
    qubit_12 = Qubit.from_value(12, length=4)
    qubit_13 = Qubit.from_value(13, length=4)
    qubit_14 = Qubit.from_value(14, length=4)
    qubit_15 = Qubit.from_value(15, length=4)

    assert gate(qubit_0) == qubit_0
    assert gate(qubit_1) == qubit_8
    assert gate(qubit_2) == qubit_2
    assert gate(qubit_3) == qubit_10
    assert gate(qubit_4) == qubit_4
    assert gate(qubit_5) == qubit_12
    assert gate(qubit_6) == qubit_6
    assert gate(qubit_7) == qubit_14
    assert gate(qubit_8) == qubit_1
    assert gate(qubit_9) == qubit_9
    assert gate(qubit_10) == qubit_3
    assert gate(qubit_11) == qubit_11
    assert gate(qubit_12) == qubit_5
    assert gate(qubit_13) == qubit_13
    assert gate(qubit_14) == qubit_7
    assert gate(qubit_15) == qubit_15

def test_swap_gate_4():
    gate = Gate.Swap(4)
    qubit_0 = Qubit.from_value(0, length=5)
    qubit_1 = Qubit.from_value(1, length=5)
    qubit_15 = Qubit.from_value(15, length=5)
    qubit_16 = Qubit.from_value(16, length=5)
    qubit_30 = Qubit.from_value(30, length=5)

    assert gate(qubit_0) == qubit_0
    assert gate(qubit_1) == qubit_16
    assert gate(qubit_15) == qubit_30
    assert gate(qubit_16) == qubit_1
    assert gate(qubit_30) == qubit_15
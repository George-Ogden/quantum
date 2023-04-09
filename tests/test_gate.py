import jax.numpy as jnp

from src.gate import *
from src.qubit import Qubit


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

def test_pauli_x_gate():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    gate = pauli_x
    assert gate(qubit_0) == qubit_1
    assert gate(qubit_1) == qubit_0

def test_hadamard_gate():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    gate = hadamard
    assert gate(qubit_0) == qubit_plus
    assert gate(qubit_1) == qubit_minus

def test_identity_gate():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    gate = identity
    assert gate(qubit_0) == qubit_0
    assert gate(qubit_1) == qubit_1

def test_neutral_gate():
    gate = hadamard
    assert gate + neutral == gate
    assert neutral + gate == gate

def test_cnot_gate_0_1_basis():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    gate = CNOT
    assert gate(qubit_0 + qubit_0) == qubit_0 + qubit_0
    assert gate(qubit_0 + qubit_1) == qubit_0 + qubit_1
    assert gate(qubit_1 + qubit_0) == qubit_1 + qubit_1
    assert gate(qubit_1 + qubit_1) == qubit_1 + qubit_0

def test_cnot_gate_plus_minus_basis():
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    gate = CNOT
    assert gate(qubit_plus + qubit_plus) == qubit_plus + qubit_plus
    assert gate(qubit_plus + qubit_minus) == qubit_minus + qubit_minus
    assert gate(qubit_minus + qubit_plus) == qubit_minus + qubit_plus
    assert gate(qubit_minus + qubit_minus) == qubit_plus + qubit_minus
import jax.numpy as jnp
import numpy as np
import pytest

from src.gate import *
from src.qubit import Qubit


def test_gate_creation_no_name():
    mat = jnp.array([[1, 0], [0, 1]])
    g = Gate(mat)
    assert jnp.allclose(g.matrix, mat)

def test_gate_creation_with_name():
    mat = jnp.array([[1, 0], [0, 1]])
    g = Gate(mat, "I")
    assert g.name == "I"
    assert jnp.allclose(g.matrix, mat)

def test_gate_call():
    mat = jnp.array([[0, 1], [1, 0]])
    q = Qubit(jnp.array([1, 0]), "q")
    g = Gate(mat)
    new_q = g(q)
    assert jnp.allclose(new_q.vector, jnp.array([0, 1]))

def test_gate_gate_mat_multiplication():
    g1 = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    g2 = Gate(jnp.array([[1, 0], [0, -1]]), "Z")
    g3 = g1 * g2
    expected = jnp.array([[0, -1], [1, 0]])
    assert jnp.allclose(g3.matrix, expected)

def test_gate_qubit_mat_multiplication():
    q = Qubit(jnp.array([1, 0]), "q")
    g = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    new_q = g @ q
    expected = jnp.array([0, 1])
    assert jnp.allclose(new_q.vector, expected)

def test_gate_multiplication():
    g1 = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    g2 = Gate(jnp.array([[1, 0], [0, -1]]), "Z")
    g3 = g1 * g2
    expected = jnp.array([[0, -1], [1, 0]])
    assert jnp.allclose(g3.matrix, expected)

def test_gate_gate_addition():
    g1 = Gate(jnp.array([[1, 0], [0, 1]]), "I")
    g2 = Gate(jnp.array([[0, 1], [1, 0]]), "X")
    g3 = g1 + g2
    expected = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert jnp.allclose(g3.matrix, expected)

def test_gate_repr(capsys):
    g = Gate(jnp.array([[1, 0], [0, 1]]), "gate_name")
    print(g)
    captured = capsys.readouterr()
    assert 'gate_name' in captured.out.lower()

def test_single_bit_Identity():
    g = Gate.Identity(1)
    assert jnp.allclose(g.matrix, jnp.array([[1, 0], [0, 1]]))

def test_two_bit_Identity():
    g = Gate.Identity(2)
    expected = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert jnp.allclose(g.matrix, expected)

def test_three_bit_Identity():
    g = Gate.Identity(3)
    expected = jnp.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ])
    assert jnp.allclose(g.matrix, expected)

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
    assert jnp.allclose(gate(qubit_0).measure(qubit_plus), 1)
    assert jnp.allclose(gate(qubit_1).measure(qubit_minus), 1)

def test_identity_gate():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    gate = identity
    assert jnp.allclose(gate(qubit_0).measure(0), 1)
    assert jnp.allclose(gate(qubit_1).measure(1), 1)

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
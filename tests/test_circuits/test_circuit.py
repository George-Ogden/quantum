import jax.numpy as jnp

from quantum.circuits.circuit import Circuit
from quantum.qubit import Qubit
from quantum.gates import *

def test_basic_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    gate = Gate.Identity(1)
    circuit = Circuit([gate])
    assert circuit(qubit_0) == qubit_0
    assert circuit(qubit_1) == qubit_1
    assert circuit(qubit_plus) == qubit_plus
    assert circuit(qubit_minus) == qubit_minus

def test_pauli_x_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    gate = Pauli_X
    circuit = Circuit([gate])
    assert circuit(qubit_0) == qubit_1
    assert circuit(qubit_1) == qubit_0

def test_two_gate_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    gate = Pauli_X
    circuit = Circuit([gate, gate])
    assert circuit(qubit_0) == qubit_0
    assert circuit(qubit_1) == qubit_1

def test_hadamard_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    gate = Hadamard
    circuit = Circuit([gate])
    assert circuit(qubit_0) == qubit_plus
    assert circuit(qubit_1) == qubit_minus

def test_serial_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    hadamard_circuit = Circuit([Hadamard])
    pauli_x_circuit = Circuit([Pauli_X])
    circuit = pauli_x_circuit * hadamard_circuit
    assert circuit(qubit_0) == qubit_minus
    assert circuit(qubit_1) == qubit_plus

def test_parallel_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    hadamard_circuit = Circuit([Hadamard])
    pauli_x_circuit = Circuit([Pauli_X])
    circuit = pauli_x_circuit + hadamard_circuit
    assert circuit(qubit_0 + qubit_0) == qubit_1 + qubit_plus
    assert circuit(qubit_0 + qubit_1) == qubit_1 + qubit_minus
    assert circuit(qubit_1 + qubit_0) == qubit_0 + qubit_plus
    assert circuit(qubit_1 + qubit_1) == qubit_0 + qubit_minus

def test_circuit_length():
    hadamard_circuit = Circuit([Hadamard])
    pauli_x_circuit = Circuit([Pauli_X])
    circuit = Circuit([Hadamard, Pauli_X])
    assert len(circuit) == 2
    assert len(pauli_x_circuit) == 1
    assert len(hadamard_circuit) == 1

def test_circuit_repr(capsys):
    circuit = Circuit([Hadamard, Pauli_X], name="Hadamard-X")
    print(circuit)
    captured = capsys.readouterr()
    assert "Hadamard-X" in captured.out
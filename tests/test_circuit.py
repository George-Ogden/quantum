import jax.numpy as jnp
import re

from src.circuit import Circuit
from src.qubit import Qubit
from src.gate import *

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
    gate = pauli_x
    circuit = Circuit([gate])
    assert circuit(qubit_0) == qubit_1
    assert circuit(qubit_1) == qubit_0

def test_two_gate_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    gate = pauli_x
    circuit = Circuit([gate, gate])
    assert circuit(qubit_0) == qubit_0
    assert circuit(qubit_1) == qubit_1

def test_hadamard_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    gate = hadamard
    circuit = Circuit([gate])
    assert circuit(qubit_0) == qubit_plus
    assert circuit(qubit_1) == qubit_minus

def test_serial_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    hadamard_circuit = Circuit([hadamard])
    pauli_x_circuit = Circuit([pauli_x])
    circuit = pauli_x_circuit * hadamard_circuit
    assert circuit(qubit_0) == qubit_minus
    assert circuit(qubit_1) == qubit_plus

def test_parallel_circuit():
    qubit_0 = Qubit.from_value(0)
    qubit_1 = Qubit.from_value(1)
    qubit_plus = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    qubit_minus = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    hadamard_circuit = Circuit([hadamard])
    pauli_x_circuit = Circuit([pauli_x])
    circuit = pauli_x_circuit + hadamard_circuit
    assert circuit(qubit_0 + qubit_0) == qubit_1 + qubit_plus
    assert circuit(qubit_0 + qubit_1) == qubit_1 + qubit_minus
    assert circuit(qubit_1 + qubit_0) == qubit_0 + qubit_plus
    assert circuit(qubit_1 + qubit_1) == qubit_0 + qubit_minus

def test_circuit_length():
    hadamard_circuit = Circuit([hadamard])
    pauli_x_circuit = Circuit([pauli_x])
    circuit = Circuit([hadamard, pauli_x])
    assert len(circuit) == 2
    assert len(pauli_x_circuit) == 1
    assert len(hadamard_circuit) == 1

def test_circuit_repr(capsys):
    circuit = Circuit([hadamard, pauli_x])
    print(circuit)
    captured = capsys.readouterr()
    assert re.search(r"H.*X", captured.out)
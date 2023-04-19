import jax.numpy as jnp

from quantum.learn import Dataset, LearnableCircuit, LearnableGate
from quantum.qubit import Minus, One, Plus, Zero
from quantum.gates import X

def test_learnable_circuit_initialisation():
    gate = LearnableGate(1)
    circuit = LearnableCircuit([gate])
    qubit = Plus
    assert circuit(qubit) == gate(qubit)

def test_learnable_circuit_fit():
    dataset = Dataset.from_qubits([
        (qubit, X(qubit)) for qubit in [Plus, Minus, One, Zero]
    ])
    gate = LearnableGate(1)
    circuit = LearnableCircuit([gate])
    circuit.fit(dataset)
    assert jnp.allclose(gate.matrix, X.matrix, atol=.1)
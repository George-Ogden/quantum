import jax.numpy as jnp

from quantum.learn import LearnableGate
from quantum.gates import Identity
from quantum.qubit import One, Plus

def test_learnable_gate_initialisation():
    gate = LearnableGate(3)
    assert gate.n == 3

def test_gate_forward():
    gate = LearnableGate(1)
    qubit = Plus
    assert jnp.allclose(jnp.linalg.norm(gate(qubit).vector), 1)

def test_gate_entanglement():
    gate = LearnableGate(1) + Identity
    qubit = Plus + One
    assert jnp.allclose(gate(qubit).measure(bit=0), 1)

def test_gate_uniqueness():
    gate1 = LearnableGate(1)
    gate2 = LearnableGate(1)
    assert gate1 != gate2
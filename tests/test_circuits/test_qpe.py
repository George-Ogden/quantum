import jax.numpy as jnp

from src.circuits import QuantumPhaseEstimation as QPE
from src.qubit import Qubit, One, Zero
from src.gates import ControlledGate, Gate

def test_qpe_small_1():
    gate = Gate(jnp.array([[0, -1j], [1j, 0]]))
    controlled_gate = ControlledGate.from_gate(gate)
    eigenvector = Qubit(jnp.array([1, -1j]) / jnp.sqrt(2))
    qpe = QPE(2, controlled_gate)
    assert qpe(
        Qubit.entangle(
            Qubit.from_value(0, length=2),
            eigenvector,
        )
    ) == Qubit.entangle(
        One,
        Zero,
        eigenvector
    )

def test_qpe_small_2():
    gate = Gate(jnp.array([[0, -1j], [1j, 0]]))
    controlled_gate = ControlledGate.from_gate(gate)
    eigenvector = Qubit(jnp.array([1, 1j]) / jnp.sqrt(2))
    qpe = QPE(2, controlled_gate)
    assert qpe(
        Qubit.entangle(
            Qubit.from_value(0, length=2),
            eigenvector,
        )
    ) == Qubit.entangle(
        Zero,
        Zero,
        eigenvector
    )

def test_qpe_small_3():
    gate = Gate(jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]]))
    controlled_gate = ControlledGate.from_gate(gate)
    eigenvector = Qubit(jnp.array([0, 1]))
    qpe = QPE(3, controlled_gate)
    print(qpe)
    print(
        qpe(Zero + Zero + Zero + eigenvector)
    )
    print(Zero + Zero + One + eigenvector)
    assert qpe(
        Qubit.entangle(
            Qubit.from_value(0, length=3),
            eigenvector,
        )
    ) == Qubit.entangle(
        Zero,
        Zero,
        One,
        eigenvector
    )

def test_qpe_medium_1():
    gate = Gate(jnp.array(
        [[(1.2071067811865475+0.49999999999999994j), (-0.20710678118654752-0.49999999999999994j), (-0.49999999999999994+0.20710678118654752j), (2.2071067811865475+0.49999999999999994j)], [(1+0j), 0j, (-1+0j), (1+0j)], [(0.20710678118654752-0.5j), (-0.20710678118654752+0.5j), (0.5+1.2071067811865475j), (0.20710678118654752-0.5j)], [0j, 0j, 0j, (-1+0j)]]
    ))
    controlled_gate = ControlledGate.from_gate(gate)
    eigenvector = Qubit(jnp.array([1, 1, 0, 0]) / jnp.sqrt(2))
    qpe = QPE(3, controlled_gate)
    assert qpe(
        Qubit.entangle(
            Qubit.from_value(0, length=3),
            eigenvector,
        )
    ) == Qubit.entangle(
        Zero,
        Zero,
        Zero,
        eigenvector
    )

def test_qpe_medium_2():
    gate = Gate(jnp.array(
        [[(1.2071067811865475+0.49999999999999994j), (-0.20710678118654752-0.49999999999999994j), (-0.49999999999999994+0.20710678118654752j), (2.2071067811865475+0.49999999999999994j)], [(1+0j), 0j, (-1+0j), (1+0j)], [(0.20710678118654752-0.5j), (-0.20710678118654752+0.5j), (0.5+1.2071067811865475j), (0.20710678118654752-0.5j)], [0j, 0j, 0j, (-1+0j)]]
    ))
    controlled_gate = ControlledGate.from_gate(gate)
    eigenvector = Qubit(jnp.array([1, 0, 1, 0]) / jnp.sqrt(2))
    qpe = QPE(3, controlled_gate)
    assert qpe(
        Qubit.entangle(
            Qubit.from_value(0, length=3),
            eigenvector,
        )
    ) == Qubit.entangle(
        Zero,
        Zero,
        One,
        eigenvector
    )

def test_qpe_medium_3():
    gate = Gate(jnp.array(
        [[(1.2071067811865475+0.49999999999999994j), (-0.20710678118654752-0.49999999999999994j), (-0.49999999999999994+0.20710678118654752j), (2.2071067811865475+0.49999999999999994j)], [(1+0j), 0j, (-1+0j), (1+0j)], [(0.20710678118654752-0.5j), (-0.20710678118654752+0.5j), (0.5+1.2071067811865475j), (0.20710678118654752-0.5j)], [0j, 0j, 0j, (-1+0j)]]
    ))
    controlled_gate = ControlledGate.from_gate(gate)
    eigenvector = Qubit(jnp.array([1, 0, 0, -1]) / jnp.sqrt(2))
    qpe = QPE(3, controlled_gate)
    assert qpe(
        Qubit.entangle(
            Qubit.from_value(0, length=3),
            eigenvector,
        )
    ) == Qubit.entangle(
        One,
        Zero,
        Zero,
        eigenvector
    )

def test_qpe_medium_4():
    gate = Gate(jnp.array(
        [[(1.2071067811865475+0.49999999999999994j), (-0.20710678118654752-0.49999999999999994j), (-0.49999999999999994+0.20710678118654752j), (2.2071067811865475+0.49999999999999994j)], [(1+0j), 0j, (-1+0j), (1+0j)], [(0.20710678118654752-0.5j), (-0.20710678118654752+0.5j), (0.5+1.2071067811865475j), (0.20710678118654752-0.5j)], [0j, 0j, 0j, (-1+0j)]]
    ))
    controlled_gate = ControlledGate.from_gate(gate)
    eigenvector = Qubit(jnp.array([0, -1, 1j, 0]) / jnp.sqrt(2))
    qpe = QPE(3, controlled_gate)
    assert qpe(
        Qubit.entangle(
            Qubit.from_value(0, length=3),
            eigenvector,
        )
    ) == Qubit.entangle(
        Zero,
        One,
        Zero,
        eigenvector
    )

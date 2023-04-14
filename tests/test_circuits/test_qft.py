from itertools import product
import jax.numpy as jnp
import numpy as np

from quantum.circuits import QuantumFourierTransform as QFT, InverseQuantumFourierTransform as IQFT
from quantum.qubit import Qubit

def test_qft_2():
    qft = QFT(2)
    for js in product([0, 1], repeat=2):
        qubit = Qubit.entangle(*[Qubit.from_value(j, length=1) for j in js])
        assert qft(qubit) == Qubit.entangle(
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * js[1])]) / jnp.sqrt(2)),
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * (js[1] / 2 + js[0]))]) / jnp.sqrt(2)),
        )

def test_qft_3():
    qft = QFT(3)
    for js in product([0, 1], repeat=3):
        qubit = Qubit.entangle(*[Qubit.from_value(j, length=1) for j in js])
        assert qft(qubit) == Qubit.entangle(
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * js[2])]) / jnp.sqrt(2)),
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * (js[2] / 2 + js[1]))]) / jnp.sqrt(2)),
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * (js[2] / 4 + js[1] / 2 + js[0]))]) / jnp.sqrt(2))
        )

def test_qft_4():
    qft = QFT(4)
    for js in product([0, 1], repeat=4):
        qubit = Qubit.entangle(*[Qubit.from_value(j, length=1) for j in js])
        assert qft(qubit) == Qubit.entangle(
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * js[3])]) / jnp.sqrt(2)),
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * (js[3] / 2 + js[2]))]) / jnp.sqrt(2)),
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * (js[3] / 4 + js[2] / 2 + js[1]))]) / jnp.sqrt(2)),
            Qubit(jnp.array([1, jnp.exp(jnp.pi * 1j * (js[3] / 8 + js[2] / 4 + js[1] / 2 + js[0]))]) / jnp.sqrt(2))
        )

def test_iqft_2():
    qft = QFT(2)
    iqft = IQFT(2)
    for i in range(4):
        qubit = Qubit.from_value(i, length=2)
        assert iqft(qft(qubit)) == qubit

def test_iqft_3():
    qft = QFT(3)
    iqft = IQFT(3)
    for i in range(8):
        qubit = Qubit.from_value(i, length=3)
        assert iqft(qft(qubit)) == qubit

def test_iqft_4():
    qft = QFT(4)
    iqft = IQFT(4)
    for i in range(16):
        qubit = Qubit.from_value(i, length=4)
        assert iqft(qft(qubit)) == qubit

def test_iqft_complex():
    qft = QFT(4)
    iqft = IQFT(4)
    qubit = Qubit(jnp.array([jnp.exp(1j * np.random.uniform(0, 2 * np.pi)) for _ in range(16)]) / 4)
    assert iqft(qft(qubit)) == qubit
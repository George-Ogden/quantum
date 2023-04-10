from itertools import product
import jax.numpy as jnp

from src.circuits import QuantumFourierTransform as QFT
from src.qubit import Qubit

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
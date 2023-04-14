import jax.numpy as jnp

from quantum.algorithms import DeutschJoszaAlgorithm
from quantum.gates import Oracle

def test_deutsch_jozsa_algorithm_balanced_1():
    table = [0, 0, 1, 1]
    oracle = Oracle.from_table(table)
    algorithm = DeutschJoszaAlgorithm(oracle)
    assert jnp.allclose(algorithm(), 1.)

def test_deutsch_jozsa_algorithm_balanced_2():
    table = [1, 0, 0, 1]
    oracle = Oracle.from_table(table)
    algorithm = DeutschJoszaAlgorithm(oracle)
    assert jnp.allclose(algorithm(), 1.)

def test_deutsch_jozsa_algorithm_constant_1():
    table = [0, 0, 0, 0]
    oracle = Oracle.from_table(table)
    algorithm = DeutschJoszaAlgorithm(oracle)
    assert jnp.allclose(algorithm(), 0.)

def test_deutsch_jozsa_algorithm_constant_2():
    table = [1, 1, 1, 1]
    oracle = Oracle.from_table(table)
    algorithm = DeutschJoszaAlgorithm(oracle)
    assert jnp.allclose(algorithm(), 0.)
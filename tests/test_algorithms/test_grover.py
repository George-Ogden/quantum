import numpy as np

from src.algorithms import GroversAlgorithm
from src.oracle import Oracle

def test_grover_small_0():
    table = [1, 0, 0, 0]
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [0., 0.], atol=.1)

def test_grover_small_1():
    table = [0, 1, 0, 0]
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [0., 1.], atol=.1)

def test_grover_small_2():
    table = [0, 0, 1, 0]
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [1., 0.], atol=.1)

def test_grover_small_3():
    table = [0, 0, 0, 1]
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [1., 1.], atol=.1)

def test_grover_medium_1():
    table = [0, 1, 0, 0, 0, 0, 0, 0]
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [0, 0, 1], atol=.1)

def test_grover_medium_5():
    table = [0, 0, 0, 0, 0, 1, 0, 0]
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [1, 0, 1], atol=.1)

def test_grover_medium_7():
    table = [0, 0, 0, 0, 0, 0, 0, 7]
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [1, 1, 1], atol=.1)

def test_grover_large_24():
    table = [0] * 64
    table[24] = 1
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [0, 1, 1, 0, 0, 0,], atol=.1)

def test_grover_large_42():
    table = [0] * 64
    table[42] = 1
    oracle = Oracle.from_table(table)
    algorithm = GroversAlgorithm(oracle)
    assert np.allclose(algorithm(), [1, 0, 1, 0, 1, 0,], atol=.1)
from quantum.gates import Oracle
from quantum.qubit import Qubit

import jax.numpy as jnp

def test_oracle_construction_from_table():
    table = [0, 0, 1, 1]
    oracle = Oracle.from_table(table)
    assert oracle == jnp.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])

def test_oracle_values_from_table():
    table = [1, 0, 0, 1]
    oracle = Oracle.from_table(table)
    for i in range(4):
        qubit = Qubit.from_value(i, length=2)
        assert oracle(qubit + Qubit.from_value(0)) == qubit + Qubit.from_value(table[i])
        assert oracle(qubit + Qubit.from_value(1)) == qubit + Qubit.from_value(1 - table[i])
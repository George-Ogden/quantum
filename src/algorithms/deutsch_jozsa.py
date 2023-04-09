import jax.numpy as jnp

from src.oracle import Oracle
from src.gate import *
from src.qubit import Qubit

def DeutschJoszaAlgorithm(oracle: Oracle):
    n = oracle.n
    x = Qubit.from_value(0, n, "x")
    y = Qubit.from_value(0, 1, "y")

    large_hadamard = sum([hadamard for _ in range(n)], start=neutral)

    # prepare states
    x, y = large_hadamard @ x, hadamard @ y
    entangled_state = x + y

    # run entangled state through oracle
    output_state = oracle(entangled_state)

    # measure the first qubit and return the result
    return output_state.measure(basis=jnp.array([1, -1]) / jnp.sqrt(2), bit=0)

if __name__ == "__main__":
    result = DeutschJoszaAlgorithm(Oracle.from_table([0,0,1,1]))  # should return 1
    print(result)
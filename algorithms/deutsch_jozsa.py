import jax.numpy as jnp
import jax
import numpy as np

from src.oracle import Oracle
from src.gates import *
from src.qubit import Qubit

n = 2
N = 2 ** n
def deutsch_algorithm(oracle: Oracle):
    x = Qubit.from_value(0, n, "x")
    y = Qubit.from_value(0, 1, "y")

    large_hadamard = sum([hadamard for _ in range(n)], start=neutral)

    # prepare states
    x, y = large_hadamard @ x, hadamard @ y
    entangled_state = x + y

    # set up unitary oracle
    output_state = oracle(entangled_state)

    # output states
    output = (large_hadamard + identity) @ output_state

    # measure the first qubit and return the result
    return output_state.measure(basis=jnp.array([1, -1]) / jnp.sqrt(2), bit=0)

if __name__ == "__main__":
    result = deutsch_algorithm(Oracle.from_table([0,0,1,1]))  # should return 1
    print(result)
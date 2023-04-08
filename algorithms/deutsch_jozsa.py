import jax.numpy as jnp
import jax
import numpy as np

from src.oracle import Oracle
from src.gates import *

n = 2
N = 2 ** n
def deutsch_algorithm(oracle: Oracle):
    x = jnp.array((1,) + (0,) * (N - 1), dtype=np.float32)
    y = jnp.array([0, 1], dtype=np.float32)

    large_hadamard = sum([hadamard for _ in range(n)], start=neutral)

    # prepare states
    x, y = large_hadamard @ x, hadamard @ y
    entangled_state = jnp.kron(x, y)

    # set up unitary oracle
    output_state = oracle(entangled_state)

    # output states
    output = (large_hadamard + identity) @ output_state

    # measure the first qubit and return the result
    first_qubit = output[:2]
    measurement = jnp.linalg.norm(jnp.dot(first_qubit, jnp.array([1, -1]) / jnp.sqrt(2)))
    return measurement

if __name__ == "__main__":
    result = deutsch_algorithm(Oracle.from_table([1,1,1,0]))  # should return {'0': 1.0}
    print(result)
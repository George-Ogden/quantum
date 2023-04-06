import jax.numpy as jnp
import jax
import numpy as np

from gates import hadamard

def deutsch_algorithm(f: jnp.ndarray):
    x = jnp.array((1, 0,), dtype=np.float32)
    y = jnp.array([0, 1], dtype=np.float32)

    # prepare states
    x, y = hadamard @ x, hadamard @ y
    entangled_state = jnp.kron(x, y)

    # set up unitary oracle
    U = jnp.zeros((2, 2), dtype=np.float32)
    U = U.at[f, jnp.arange(2)].set(1)

    # pass state through oracle
    output_state = jnp.kron(jnp.eye(2), U) @ entangled_state

    # output states
    output = jnp.dot(jnp.kron(hadamard, jnp.eye(2)), output_state)

    # measure the first qubit and return the result
    first_qubit = output[::2]
    measurement = jnp.dot(first_qubit, jnp.array([1, -1]) / jnp.sqrt(2))
    return measurement


# Example usage:
result = deutsch_algorithm(np.array([1,  0]))  # should return {'0': 1.0}
print(result)
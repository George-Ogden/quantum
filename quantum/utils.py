import jax.numpy as jnp

# converts a vector to a matrix
def to_matrix(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    return x

# calculates the conjugate transpose of a matrix
def hermitian(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.conj(to_matrix(x).T)
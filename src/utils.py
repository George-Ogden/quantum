import jax.numpy as jnp

def to_matrix(x: jnp.ndarray) -> jnp.ndarray:
    if x.ndim == 1:
        x = x[:, jnp.newaxis]
    return x

def hermitian(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.conj(to_matrix(x).T)
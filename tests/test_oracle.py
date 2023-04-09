from src.gate import Oracle
import jax.numpy as jnp

def test_oracle():
    table = [0, 0, 1, 1]
    oracle = Oracle.from_table(table)
    assert oracle.matrix.shape == (8, 8)
    for i in range(4):
        x = jnp.zeros(4, dtype=int)
        x = x.at[i].set(1)
        f_x = jnp.kron(x, jnp.array([1, 0]))
        f_xor = jnp.kron(x, jnp.array([0, 1]))
        print(f_x, oracle(f_x))
        print(f_xor, oracle(f_xor))
        # if table[i] == 0:
        #     assert jnp.allclose(oracle(f_x), f_x)
        #     assert jnp.allclose(oracle(f_xor), f_xor)
        # else:
        #     assert jnp.allclose(oracle(f_x), f_xor)
        #     assert jnp.allclose(oracle(f_xor), f_x)
    assert False

def test_gates():
    
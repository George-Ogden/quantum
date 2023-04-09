import jax.numpy as jnp

from src.qubit import Qubit

def test_single_bit_0():
    qubit = Qubit.from_value(0)
    assert jnp.allclose(qubit.vector, jnp.array([1, 0]))

def test_single_bit_1():
    qubit = Qubit.from_value(1)
    assert jnp.allclose(qubit.vector, jnp.array([0, 1]))

def test_multi_bit_0():
    qubit = Qubit.from_value(0, length=2)
    assert jnp.allclose(qubit.vector, jnp.array([1, 0, 0, 0]))

def test_multi_bit_3():
    qubit = Qubit.from_value(3, length=2)
    assert jnp.allclose(qubit.vector, jnp.array([0, 0, 0, 1]))

def test_multi_bit_5():
    qubit = Qubit.from_value(5, length=3)
    assert jnp.allclose(qubit.vector, jnp.array([0, 0, 0, 0, 0, 1, 0, 0]))

def test_superposition_10():
    qubit_1 = Qubit.from_value(1, length=1)
    qubit_0 = Qubit.from_value(0, length=1)
    assert qubit_1 + qubit_0 == Qubit.from_value(2, length=2)

def test_superposition_101():
    qubit_2 = Qubit.from_value(2, length=2)
    qubit_1 = Qubit.from_value(1, length=1)
    assert qubit_2 + qubit_1 == Qubit.from_value(5, length=3)

def test_qubit_repr(capsys):
    qubit = Qubit(jnp.array([1, 0]), name="test_qubit")
    print(qubit)
    captured = capsys.readouterr()
    assert "test_qubit" in captured.out.lower()

def test_repr_from_value(capsys):
    qubit = Qubit.from_value(0, name="test_qubit")
    print(qubit)
    captured = capsys.readouterr()
    assert "test_qubit" in captured.out

def test_measure_0():
    qubit = Qubit.from_value(0)
    assert jnp.allclose(qubit.measure(basis=jnp.array([1, 0])), 1)
    assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1])), 0)

def test_measure_1():
    qubit = Qubit.from_value(1)
    assert jnp.allclose(qubit.measure(basis=jnp.array([1, 0])), 0)
    assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1])), 1)

def test_measure_plus():
    qubit = Qubit(jnp.array([1, 1]) / jnp.sqrt(2))
    assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1])), .5)
    assert jnp.allclose(qubit.measure(basis=jnp.array([1, 0])), .5)

def test_measure_minus():
    qubit = Qubit(jnp.array([1, -1]) / jnp.sqrt(2))
    assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1])), .5)
    assert jnp.allclose(qubit.measure(basis=jnp.array([1, 0])), .5)

def test_measure_multiple_bits():
    qubit = Qubit(jnp.array([0, .5, .5, .5, .5, 0, 0, 0]))
    assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1]), bit=0), .5)
    assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1]), bit=1), .5)
    assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1]), bit=2), .25)

def test_measure_large_basis():
    qubit = Qubit.from_value(42, length=6)
    for i in range(6):
        assert jnp.allclose(qubit.measure(basis=jnp.array([0, 1]), bit=i), i % 2)

def test_measure_multiple_bits_entangled():
    qubit = Qubit(jnp.array([.25] * 16))
    plus_basis = jnp.array([1, 1]) / jnp.sqrt(2)
    minus_basis = jnp.array([1, -1]) / jnp.sqrt(2)
    for i in range(4):
        assert jnp.allclose(qubit.measure(basis=plus_basis, bit=i), 1)
        assert jnp.allclose(qubit.measure(basis=minus_basis, bit=i), 0)
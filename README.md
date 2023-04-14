# Quantum Library
This is a library for simulating quantum computers written in JAX.
## Why JAX?
It's **fast** and *differentiable*. I'm still thinking of a way that that would be useful.
## Components
- Qubits
- Gates
- Circuits
- Algorithms
### Qubits
```python
>>> from quantum import qubit
>>> import jax.numpy as jnp
>>> plus = qubit.Qubit(jnp.array([1,1]) / jnp.sqrt(2), name="Plus")
>>> plus # display
plus [0.71 0.71]
>>> plus == qubit.Plus # equality
Array(True, dtype=bool)
>>> plus.measure()
Array(0.5, dtype=float32)
>>> one = qubit.Qubit.from_value(1) # create from value
>>> zero = qubit.Qubit.from_value(0)
>>> one + zero == qubit.Qubit.from_value(2, length=2) # entanglement
Array(True, dtype=bool)
>>> qubit.Qubit.entangle(one, zero, zero) # multiple entanglement
qubit + qubit + qubit [0. 0. 0. 0. 1. 0. 0. 0.]
>>> qubit.Qubit(jnp.array([1, 1j])) # supports complex numbers
qubit [0.71+0.j   0.  +0.71j]
```
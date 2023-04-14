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
>>> from quantum.qubit import Qubit
>>> from quantum import qubit
>>> import jax.numpy as jnp
>>> plus = Qubit(jnp.array([1,1]) / jnp.sqrt(2), name="Plus")
# display
>>> plus
plus [0.71 0.71]
# equality
>>> plus == qubit.Plus
Array(True, dtype=bool)
>>> plus.measure()
Array(0.5, dtype=float32)
# create from value
>>> one = Qubit.from_value(1)
>>> zero = Qubit.from_value(0)
# entanglement
>>> one + zero == Qubit.from_value(2, length=2)
Array(True, dtype=bool)
# multiple entanglement
>>> Qubit.entangle(one, zero, zero)
qubit + qubit + qubit [0. 0. 0. 0. 1. 0. 0. 0.]
# supports complex numbers
>>> Qubit(jnp.array([1, 1j]))
qubit [0.71+0.j   0.  +0.71j]
```
### Gates
```python
>>> from quantum.gates import Gate, Oracle, ControlledGate
>>> from quantum import gates
>>> import jax.numpy as jnp
# gate construction
>>> Gate(jnp.array([[1, 0], [0, 1j]]), name="Phase")
PHASE (2, 2)
# built-in useful gates
>>> gates.X(qubit.One) == qubit.Zero
Array(True, dtype=bool)
# built-in useful functionality
>>> Gate.Swap(2)(Qubit.from_value(4, length=3)) == Qubit.from_value(1, length=3) # swap "bits" 0 and 2
Array(True, dtype=bool)
# controlled gates
>>> controlled_hadamard = ControlledGate.from_gate(gates.Hadamard)
>>> controlled_hadamard
C-H (4, 4)
>>> controlled_hadamard(qubit.Zero + qubit.Zero) # don't apply if second qubit is 0
zero + zero [1. 0. 0. 0.]
>>> controlled_hadamard(qubit.Zero + qubit.One) # apply if second qubit is one
zero + one [0.   0.71 0.   0.71]
>>> Gate.Identity(3)(Qubit.from_value(4, length=3))
qubit [0. 0. 0. 0. 1. 0. 0. 0.]
# serial composition
>>> (gates.X * gates.X)(qubit.One) == qubit.One
Array(True, dtype=bool)
>>> Gate.serial(gates.T, gates.T, gates.S) == gates.Z
Array(True, dtype=bool)
# parallel composition
>>> (gates.X + gates.X)(qubit.One + qubit.Zero) == qubit.Zero + qubit.One
Array(True, dtype=bool)
>>> parallel = Gate.parallel(gates.X, gates.Y, gates.Z)
>>> parallel
((X + Y) + Z) (8, 8)
>>> parallel(qubit.Zero + qubit.Zero + qubit.Zero)
zero + zero + zero [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+1.j 0.+0.j]
# oracle
>>> oracle = Oracle.from_table([0, 1, 0, 1])
>>> oracle
Oracle (8, 8)
>>> oracle(Qubit.from_value(2, length=2) + qubit.Zero).measure(0)
Array(1., dtype=float32)
>>> oracle(Qubit.from_value(2, length=2) + qubit.One).measure(0)
Array(0., dtype=float32)
# accepts entangled qubits in a superposition
>>> oracle(qubit.Plus + qubit.Plus + qubit.One).measure(0)
Array(0.5, dtype=float32)
```
### Algorithms
```python
from quantum.algorithms import DeutschJoszaAlgorithm, GroversAlgorithm
# Deutsch-Jozsa Algorithm
>>> oracle = Oracle.from_table([0, 1, 0, 1])
>>> algorithm = DeutschJoszaAlgorithm(oracle)
>>> algorithm.run()
[Array(-0., dtype=float32), Array(1., dtype=float32)] # constant in "bit" 1 and balanced in "bit" 0
>>> oracle = Oracle.from_table([0, 1, 1, 0])
>>> algorithm = DeutschJoszaAlgorithm(oracle)
>>> algorithm.run()
[Array(1., dtype=float32), Array(1., dtype=float32)] # balanced in both "bits"
>>> oracle = Oracle.from_table([0, 0, 0, 0])
>>> algorithm = DeutschJoszaAlgorithm(oracle)
>>> algorithm.run()
[Array(-0., dtype=float32), Array(-0., dtype=float32)] # constant in both "bits"
# Grover's Algorithm
>>> oracle = Oracle.from_table([0, 1, 0, 0])
>>> algorithm = GroversAlgorithm(oracle)
>>> algorithm.run()
[Array(0., dtype=float32), Array(1., dtype=float32)] # 1st value is a 1
>>> oracle = Oracle.from_table([0, 0, 0, 0, 0, 0, 1, 0])
>>> algorithm = GroversAlgorithm(oracle)
>>> algorithm.run()
[Array(0.97, dtype=float32), Array(0.97, dtype=float32), Array(0.03, dtype=float32)] # 6th value is a 1
```
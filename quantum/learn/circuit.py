from collections import defaultdict
import jax.numpy as jnp
import optax
import jax

from typing import Dict, List, Optional

from ..circuits.circuit import Circuit
from ..utils import hermitian
from ..qubit import Qubit
from ..gates import Gate

from .gate import LearnableGate
from .dataset import Dataset

class LearnableCircuit(Circuit):
    def __init__(self, gates: List[Gate], name: Optional[str] = None):
        super().__init__(gates, name)
        self.learnable_gates = {gate.idx: gate for gate in gates if isinstance(gate, LearnableGate)}

    def grad_func(self, gates: Dict[int, jnp.array], x: jnp.array, y: jnp.array) -> float:
        """Returns the gradient of the loss function with respect to the learnable gates"""
        self.swap_gates(gates)
        target_loss = self.compute_loss(x, y)
        self.swap_gates(gates)
        n = len(x)
        unitary_loss = sum(jnp.linalg.norm(jnp.eye(n) - (hermitian(gate) @ gate)) + jnp.linalg.norm(jnp.eye(n) - (gate @ hermitian(gate))) for gate in gates.values())
        return target_loss * 1 + unitary_loss * .01
 
    def compute_loss(self, input: jnp.array, target: jnp.array) -> float:
        """Computes the loss function for the given qubit"""
        output = self(Qubit(input))
        loss = jnp.linalg.norm(output.vector - target)
        return loss            

    def swap_gates(self, gates: Dict[int, jnp.array]):
        for k in gates:
            current_matrix = self.learnable_gates[k].matrix
            self.learnable_gates[k].matrix = gates[k]
            gates[k] = current_matrix

    def fit(self, dataset: Dataset):
        """Fits the learnable gates in the circuit to the dataset"""
        optimizer = optax.adam(1e-2)
        params = {k: v.matrix for k, v in self.learnable_gates.items()}
        opt_state = optimizer.init(params)
        loss_and_grad = jax.jit(jax.value_and_grad(self.grad_func))

        for epoch in range(1000):
            losses = 0
            grads = defaultdict(int)
            for x, y in dataset:
                loss, grad = loss_and_grad(params, x.vector, y.vector)
                grads = {k: grads[k] + grad[k] for k in grad}
                losses += loss
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            print(f"Epoch {epoch}: {losses / len(dataset)}")
        self.swap_gates(params)
        idx, = self.learnable_gates.keys()
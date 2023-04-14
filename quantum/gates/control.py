from __future__ import annotations

import jax.numpy as jnp

from typing import Optional

from .gate import Gate

class ControlledGate(Gate):
    """a gate with control qubits"""
    def __init__(self, matrix: jnp.ndarray, name: Optional[str] = None, target: int = 1):
        """creates a controlled gate from a matrix with control qubits

        Args:
            matrix (jnp.ndarray): original gate
            name (Optional[str], optional): name of the gate. Defaults to None.
            target (int, optional): number of control qubits. Defaults to 1.
        """
        super().__init__(matrix, name)
        self.t = target

    @property
    def n(self) -> int:
        """return number of qubits the gate operates on (excluding) control qubits"""
        return super().n - self.t

    @staticmethod
    def from_gate(gate: Gate, t: int = 1) -> ControlledGate:
        """creates a controlled gate from a gate with t control qubits"""
        # modified from https://github.com/Qiskit/qiskit-terra/blob/3cf63baa3582d6cd5bcbeb976659dde3236f9007/qiskit/extensions/unitary.py#L130
        control_projection = jnp.diag(
            jnp.roll(
                jnp.repeat(
                    jnp.array([[1], [0]]),
                    jnp.array([1, 2 ** t - 1])
                ),
                2 ** t - 1
            )
        )
        matrix = jnp.kron(
            jnp.eye(2 ** gate.n),
            jnp.eye(2 ** t) - control_projection
        ) + jnp.kron(
            gate.matrix,
            control_projection
        )
        return ControlledGate(matrix, f"C-{gate.name}", t)
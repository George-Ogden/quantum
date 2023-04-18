from __future__ import annotations

import jax.numpy as jnp

from typing import List, Iterator, Tuple

from ..qubit import Qubit

class Dataset:
    def __init__(self, data: jnp.array):
        """_summary_

        Args:
            data (jnp.array): a jnp.array of shape (N, 2, 2^n) where n is the number of qubits and N is the number of samples
        """
        if not isinstance(data, jnp.ndarray):
            data = jnp.array(data)
        assert data.ndim == 3 and data.shape[1] == 2, "data must be of shape (N, 2, 2^n)"
        self.data = data
    
    def __getitem__(self, index: int) -> Tuple[Qubit, Qubit]:
        return Qubit(self.data[index, 0]), Qubit(self.data[index, 1])

    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> Iterator[Tuple[Qubit, Qubit]]:
        for i in range(len(self)):
            yield self[i]
    
    def from_qubits(qubits: List[Tuple[Qubit, Qubit]]) -> Dataset:
        """Creates a dataset from a list of qubits

        Args:
            qubits (List[Tuple[Qubit, Qubit]]): a list of qubits

        Returns:
            Dataset: a dataset
        """
        return Dataset(
            jnp.array([
                [qubit1.vector, qubit2.vector]
                for qubit1, qubit2 in qubits
            ])
        )
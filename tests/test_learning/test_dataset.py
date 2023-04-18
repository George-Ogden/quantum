import jax.numpy as jnp

from quantum.learn import Dataset
from quantum.qubit import Qubit

test_dataset = Dataset(
    jnp.stack(
        (
            jnp.eye(8),
            jnp.roll(jnp.eye(8), 1, axis=1)
        ),
        axis=1
    )
)

def test_dataset_initialisation_from_qubits():
    dataset = Dataset.from_qubits(
        [
            (
                Qubit.from_value(i, 3),
                Qubit.from_value((i + 1)%8, 3)
            ) for i in range(8)
        ]
    )
    assert (dataset.data == test_dataset.data).all()

def test_dataset_length():
    assert len(test_dataset) == 8

def test_dataset_iteration():
    for i, (qubit1, qubit2) in enumerate(test_dataset):
        assert qubit1 == Qubit.from_value(i, 3)
        assert qubit2 == Qubit.from_value((i + 1)%8, 3)

def test_dataset_indexing():
    for i in range(8):
        qubit1, qubit2 = test_dataset[i]
        assert qubit1 == Qubit.from_value(i, 3)
        assert qubit2 == Qubit.from_value((i + 1)%8, 3)
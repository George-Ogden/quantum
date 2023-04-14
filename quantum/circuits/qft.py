from ..gates import Gate, Hadamard, Identity

from .circuit import Circuit

class QuantumFourierTransform(Circuit):
    """Quantum Fourier Transform (QFT) Circuit
    find "frequencies" of the quits
    for more information see: https://www.cl.cam.ac.uk/teaching/2223/QuantComp/Quantum_Computing_Lecture_9_2023.pdf
    """
    def __init__(self, n: int):
        """create a QFT circuit to find the frequencies of n qubits

        Args:
            n (int): number of input qubits
        """
        gates = []
        # apply hadamard and R gates to each qubit
        for i in range(n):
            # apply hadamard gate to this qubit
            gates.append(Gate.Identity(i) + Hadamard + Gate.Identity(n - i - 1))
            for j in range(i + 1, n):
                # swap the neighbouring qubit with one further away
                swap = Identity + (Gate.Swap(j - i - 1) if j > i + 1 else Gate.Identity(1))
                # apply R gate to this qubit and the swapped one
                gates.append(Gate.Identity(i) + (swap * (Gate.CROT(j - i + 1) + Gate.Identity(j - i - 1)) * swap) + Gate.Identity(n - j - 1))
        
        # swap the qubits back to their original order
        for i in range(n // 2):
            gates.append(Gate.Identity(i) + Gate.Swap(n - 2 * i - 1) + Gate.Identity(i))
        super().__init__(gates, "QFT")

class InverseQuantumFourierTransform(Circuit):
    """Inverse Quantum Fourier Transform (IQFT) Circuit
    convert frequencies back to qubits (inverse of QFT)
    for more information see: https://www.cl.cam.ac.uk/teaching/2223/QuantComp/Quantum_Computing_Lecture_9_2023.pdf
    """
    def __init__(self, n: int):
        """create an IQFT circuit to convert n frequencies back to qubits

        Args:
            n (int): number of input qubits
        """
        qft = QuantumFourierTransform(n)
        # apply the inverse of each gate in the QFT circuit
        gates = [gate.inverse for gate in reversed(qft.gates)]
        super().__init__(gates, "IQFT")
from qiskit import QuantumCircuit
import numpy as np

class QuantumKernelCircuits:
    """
    Provides NISQ-friendly circuits.
    Focus on shallow depth and parameterized options.
    """
    
    @staticmethod
    def feature_map_circuit(n_qubits: int, data_params: np.ndarray) -> QuantumCircuit:
        """
        Creates a data encoding circuit (Angle Encoding).
        Shallow depth: O(1) layers of rotations.
        """
        qc = QuantumCircuit(n_qubits)
        # Angle encoding: Ry rotation by data values * pi
        for i in range(min(n_qubits, len(data_params))):
            qc.ry(data_params[i] * np.pi, i)
        return qc

    @staticmethod
    def model_selection_circuit(feature_vector: np.ndarray) -> QuantumCircuit:
        """
        Multi-objective model-selection circuit.
        Qubits encode endpoint choice probabilities.
        Bitstring mapping:
            00 -> High-Quality-Expensive
            01 -> Medium-Balanced
            10 -> Fast-Cheap
        """
        n_qubits = 2
        qc = QuantumCircuit(n_qubits)
        f = np.clip(feature_vector, 0.0, 1.0)
        complexity = f[1] if len(f) > 1 else 0.5
        priority = f[2] if len(f) > 2 else 0.5
        budget_constraint = f[3] if len(f) > 3 else 0.5

        # Angle encoding of prompt properties.
        qc.ry(np.pi * complexity, 0)
        qc.ry(np.pi * (1.0 - budget_constraint), 1)

        # Encourage quality path when complexity and priority are high.
        qc.rz(np.pi * priority, 0)
        qc.rz(np.pi * (1.0 - complexity), 1)
        qc.h(range(n_qubits))
        qc.cz(0, 1)
        return qc

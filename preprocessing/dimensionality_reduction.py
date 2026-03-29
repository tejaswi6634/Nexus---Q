import numpy as np


class DimensionalityReducer:
    """
    Metadata embedding reducer for quantum angle encoding.
    Maintains a fixed-size normalized vector in [0, 1].
    """

    def __init__(self, target_dim: int = 4):
        self.target_dim = target_dim

    def learn(self, data_vector: np.ndarray):
        # Kept for interface compatibility; no-op in metadata embedding mode.
        return None

    def reduce(self, data_vector: np.ndarray) -> np.ndarray:
        vec = np.array(data_vector, dtype=float).flatten()
        if len(vec) < self.target_dim:
            vec = np.pad(vec, (0, self.target_dim - len(vec)), constant_values=0.0)
        elif len(vec) > self.target_dim:
            vec = vec[: self.target_dim]
        return np.clip(vec, 0.0, 1.0)

    def update_target_dim(self, new_dim: int):
        if new_dim != self.target_dim:
            self.target_dim = new_dim

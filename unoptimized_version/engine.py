import math
import time


class FeatureEngineUnoptimized:
    """Unoptimized processing engine containing structural performance bottlenecks."""

    def compute_pairwise_distances(self, data: list[list[float]]) -> list[list[float]]:
        """Calculates Euclidean pairwise distance matrix using unvectorized triple-nested loops."""
        n = len(data)
        matrix = []

        for i in range(n):
            row = []
            for j in range(n):
                # Bottleneck 1: Explicit item-by-item calculation with repeated math ops
                dist_sum = 0.0
                dim = len(data[i])
                for k in range(dim):
                    diff = data[i][k] - data[j][k]
                    dist_sum += diff * diff
                row.append(math.sqrt(dist_sum))
            matrix.append(row)

        return matrix

    def normalize_and_filter(self, features: list[float], threshold: float) -> list[float]:
        """Normalizes dataset with redundant mean/std recalculations inside loop."""
        filtered = []

        # Bottleneck 2: Recalculating mean and std repeatedly inside the iteration loop
        for val in features:
            mean = sum(features) / len(features)
            variance = sum((x - mean) ** 2 for x in features) / len(features)
            std = math.sqrt(variance) if variance > 0 else 1.0

            z_score = (val - mean) / std
            if abs(z_score) <= threshold:
                filtered.append(val)

        return filtered

    def find_unique_ids(self, id_list: list[int]) -> list[int]:
        """Filters duplicates using inefficient linear array search."""
        unique_ids = []

        # Bottleneck 3: Linear search in list (O(N^2) time complexity)
        for item in id_list:
            if item not in unique_ids:
                unique_ids.append(item)

        return unique_ids
import math


class FeatureEngineOptimized:
    """Optimized processing engine eliminating algorithmic and computational bottlenecks."""

    def compute_pairwise_distances(self, data: list[list[float]]) -> list[list[float]]:
        """Optimized distance calculation using pre-allocated lists and flattened comprehension."""
        n = len(data)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            vec_i = data[i]
            dim = len(vec_i)
            for j in range(i + 1, n):
                vec_j = data[j]
                dist_sum = sum((vec_i[k] - vec_j[k]) ** 2 for k in range(dim))
                dist = math.sqrt(dist_sum)
                matrix[i][j] = dist
                matrix[j][i] = dist

        return matrix

    def normalize_and_filter(self, features: list[float], threshold: float) -> list[float]:
        """Optimized normalization by pre-computing mean and std ONCE outside the loop."""
        if not features:
            return []

        # Optimization: Pre-compute stats once (O(N) time complexity)
        n = len(features)
        mean = sum(features) / n
        variance = sum((x - mean) ** 2 for x in features) / n
        std = math.sqrt(variance) if variance > 0 else 1.0

        # Vectorized list comprehension filtering
        return [val for val in features if abs((val - mean) / std) <= threshold]

    def find_unique_ids(self, id_list: list[int]) -> list[int]:
        """Optimized duplicate removal using set-based hash lookup (O(N) complexity)."""
        seen = set()
        unique_ids = []

        for item in id_list:
            if item not in seen:
                seen.add(item)
                unique_ids.append(item)

        return unique_ids
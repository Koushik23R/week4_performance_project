import unittest
from unoptimized_version.engine import FeatureEngineUnoptimized
from optimized_version.engine import FeatureEngineOptimized


class TestEngineParity(unittest.TestCase):
    def setUp(self):
        self.unoptimized = FeatureEngineUnoptimized()
        self.optimized = FeatureEngineOptimized()

    def test_pairwise_distances_parity(self):
        data = [[1.0, 2.0], [4.0, 6.0], [7.0, 8.0]]
        res_unopt = self.unoptimized.compute_pairwise_distances(data)
        res_opt = self.optimized.compute_pairwise_distances(data)

        for i in range(len(data)):
            for j in range(len(data)):
                self.assertAlmostEqual(res_unopt[i][j], res_opt[i][j], places=5)

    def test_normalize_and_filter_parity(self):
        features = [10.0, 20.0, 30.0, 100.0, -50.0]
        res_unopt = self.unoptimized.normalize_and_filter(features, threshold=1.5)
        res_opt = self.optimized.normalize_and_filter(features, threshold=1.5)
        self.assertEqual(res_unopt, res_opt)

    def test_find_unique_ids_parity(self):
        ids = [10, 20, 10, 30, 20, 40]
        res_unopt = self.unoptimized.find_unique_ids(ids)
        res_opt = self.optimized.find_unique_ids(ids)
        self.assertEqual(res_unopt, res_opt)


if __name__ == "__main__":
    unittest.main()
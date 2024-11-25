import unittest
import numpy as np
from affinityPropagation import calculate_similarity, update_responsibility_availability, affinity_propagation

class TestAffinityPropagation(unittest.TestCase):
    def setUp(self):
        # Prosty zbiór danych 2D
        self.X = np.array([
            [1.0, 2.0],
            [1.1, 2.1],
            [8.0, 9.0],
            [8.1, 9.1],
        ])
        # Parametry testowe
        self.n_samples = self.X.shape[0]
        self.damping = 0.9
        self.S = calculate_similarity(self.X)
        self.R = np.zeros((self.n_samples, self.n_samples))
        self.A = np.zeros((self.n_samples, self.n_samples))

    def test_calculate_similarity(self):
        expected_similarity = np.array([
            [0, -0.02, -97.0, -97.22],
            [-0.02, 0, -96.02, -96.22],
            [-97.0, -96.02, 0, -0.02],
            [-97.22, -96.22, -0.02, 0],
        ])
        np.testing.assert_almost_equal(self.S, expected_similarity, decimal=2)

    def test_update_responsibility_availability(self):
        R, A = update_responsibility_availability(self.S, self.R, self.A, self.damping)
        self.assertEqual(R.shape, (self.n_samples, self.n_samples))
        self.assertEqual(A.shape, (self.n_samples, self.n_samples))

        # Sprawdź, czy wartości są zgodne z oczekiwanym zakresem
        self.assertTrue(np.all(R <= 0))
        self.assertTrue(np.all(A <= 0))

    def test_affinity_propagation(self):
        """Test pełnego algorytmu."""
        clusters, centers = affinity_propagation(self.X, max_iter=100, damping=0.9)
        # Sprawdź, czy liczba klastrów jest zgodna z oczekiwaniami
        self.assertGreaterEqual(len(centers), 1)
        self.assertLessEqual(len(centers), self.n_samples)

        self.assertEqual(len(clusters), self.n_samples)

if __name__ == "__main__":
    unittest.main()
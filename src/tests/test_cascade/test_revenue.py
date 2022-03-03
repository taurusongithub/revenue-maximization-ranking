import unittest

from revenue_maximization_ranking.cascade.revenue import expected_revenue
from scipy.stats import randint


class TestExpectedRevenue(unittest.TestCase):

    def test_revenue(self):
        g = randint(1, 4)
        ranking = [("A", {"revenue": 1.2, "probability": 0.1}),
                   ("B", {"revenue": 2.2, "probability": 0.01}),
                   ("C", {"revenue": 1.7, "probability": 0.05})]
        y = (0.1 * 1.2
             + (1 - 0.1) * 0.01 * 2.2 * 2 / 3
             + (1 - 0.1) * (1 - 0.01) * 0.05 * 1.7 / 3)
        x = expected_revenue(ranking, g)
        self.assertAlmostEqual(x, y, places=10,
                               msg="Expected revenue failed the example!")

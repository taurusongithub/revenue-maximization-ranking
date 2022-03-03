import unittest

from revenue_maximization_ranking.cascade\
                                 .fixed_attention import optimal_rankings


class TestOptimalRankings(unittest.TestCase):

    def test_optimal_rankings(self):
        p_a = {"revenue": 1.0, "probability": 1.0}
        p_b = {"revenue": 9.0, "probability": 0.1}
        p_c = {"revenue": 1.9, "probability": 0.52}
        products = {"a": p_a, "b": p_b, "c": p_c}
        capacity = 2
        rankings, revenues = optimal_rankings(products=products,
                                              capacity=capacity)
        for k in rankings:
            msg = "Rankings must be of length equal to the attention"
            self.assertEqual(len(rankings[k]), k, msg)

        msg = "optimal_rankings failed at k = 1."
        self.assertEqual(rankings[1], [("a", p_a)], msg)
        self.assertEqual(revenues[1], 1.0, msg)

        msg = "optimal_rankings failed at k = 2."
        self.assertEqual(rankings[2], [("b", p_b), ("a", p_a)], msg)
        self.assertEqual(revenues[2], 1.8, msg)


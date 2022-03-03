"""Revenue calculation.

Functions
----------
    expected_revenue:
        It calculates the expected revenue of a ranking.
"""

from typing import Iterable
from revenue_maximization_ranking._types import DistributionLike

__all__ = ["expected_revenue"]


def expected_revenue(ranked_products: Iterable, g: DistributionLike) -> float:
    """It calculates the expected revenue of a ranking.

    Parameters
    ----------
        ranked_products: Iterable
            An iterable object of tuples which represents products, the
            first element in the tuple should be an id and the second
            element a dictionary with the revenue and probability of
            the product.
        g: DistributionLike
            Distribution of the customers' attention spans.

    Returns
    -------
        revenue: float
            Expected revenue for the given ranking and distribution of
            attention spans.
    """

    negative_cumm_prob = 1.0
    revenue = 0.0
    for i, product in enumerate(ranked_products):
        prob = product[1]["probability"]
        revenue += (negative_cumm_prob * prob
                    * product[1]["revenue"]
                    * (g.sf(i + 1) + g.pmf(i + 1)))
        negative_cumm_prob *= (1 - prob)

    return revenue

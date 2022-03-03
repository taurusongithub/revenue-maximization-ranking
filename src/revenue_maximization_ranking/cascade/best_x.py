"""The Best-x algorithm.

When facing the random attention span problem, the optimal ranking lets
call it "sigma" for a fixed attention span x (one from {1, 2, ..., M})
make up to a revenue of R("sigma", x) for a fraction G_x = Prob(X >= x)
of customers, where X is the distribution of attention spans.

Following this strategy each possible value of x gives a lower bound on
the expected revenue, so we choose the "best x"! That is the one for
which the lower bound is maximum, the corresponding optimal ranking for
that attention span will be used.

Reference:
    "Revenue Maximization and Learning in Products Ranking"
    (https://arxiv.org/abs/2012.03800) by Ningyuan Chen, Anran Li,
    Shuoguang Yang. That's the preprint version on which this code
    is inspired, the article has already been published so here's
    the proper citation:
        Ningyuan Chen, Anran Li, and Shuoguang Yang. 2021.
    Revenue Maximization and Learning in Products Ranking.
    <i>Proceedings of the
    22nd ACM Conference on Economics and Computation</i>.
    Association for Computing Machinery, New York, NY, USA, 316–317.
    DOI:https://doi.org/10.1145/3465456.3467610

Since this "best" x can be much smaller than the capacity M we propose
the following adaptation:
  - Choose the "best" x and its corresponding ranking for the first x
  elements,
  - remove those first x products from the original set of products and
  solve for the "best next" x again considering capacity M - x.
  - repeat until capacity is 0.

Functions
---------
    best_x:
        It finds the x for the maximum lower bound on expected revenue.

    best_x_full_capacity:
        It completes the best-x strategy up to full capacity.
"""

from copy import copy
from typing import Tuple, List, Dict
from revenue_maximization_ranking.cascade\
                                 .fixed_attention import optimal_rankings
from revenue_maximization_ranking._types import DistributionLike

__all__ = ["best_x_full_capacity", "best_x"]


def best_x(products: Dict, g: DistributionLike, capacity: int,
           offset: int = 0) -> Tuple[int, List]:
    """It finds the x for the maximum lower bound on expected revenue.

    Given a set of products each fixed attention span "x" from
    1, 2, ..., min(len(products), capacity) with its corresponding
    optimal ranking gives a lower bound on the expected revenue. This
    function chooses the best x based on that lower bound.

    Parameters
    ----------
        products: dict
            Set of products, keys must be the product ids and values
            must be dictionaries with the revenue and probability of
            each product.
        g: DistributionLike
            Distribution of attention spans.
        capacity: int
            Maximum number of items that the retailer can display.
        offset: optional, int, default 0
            An offset to be used when calling the distribution g.

    Returns
    -------
        best_x_value, rankings[best_x_value]: tuple[int, list]
            The best-x and its optimal ranking.
    """

    rankings, revenues = optimal_rankings(products=products,
                                          capacity=capacity)
    maximum_lower_bound = 0.0
    best_x_value = 0
    for x, revenue_best_assort in revenues.items():
        revenue_lower_bound = revenue_best_assort * (g.sf(x + offset)
                                                     + g.pmf(x + offset))
        if revenue_lower_bound > maximum_lower_bound:
            best_x_value = x
            maximum_lower_bound = revenue_lower_bound

    if best_x_value == 0:
        # This could happen depending on the distribution g
        return 0, []

    return best_x_value, rankings[best_x_value]


def best_x_full_capacity(products: Dict, g: DistributionLike,
                         capacity: int) -> List:
    """It completes the best-x strategy up to full capacity.

    Since the "best" x can be much smaller than the capacity M this
    function implements the following adaptation:
      - Choose the "best" x and its corresponding ranking for the first
      x elements,
      - define offset as the amount of items already ranked,
      - remove those first "offset" products from the original set of
      products and solve for the "best next" x again considering
      capacity M - offset and the evaluation of distribution g with
      offset.
      - Repeat until capacity is 0.

    Parameters
    ----------
        products: dict
            Set of products, keys must be the product ids and values
            must be dictionaries with the revenue and probability of
            each product.
        g: DistributionLike
            Distribution of attention spans.
        capacity: int
            Maximum number of items that the retailer can display.

    Returns
    -------
        full_ranking:
            Ranking of products from this iterative "best-x" strategy.
            Length of this ranking is min(M, max_x) where max_x is the
            maximum attention considered by the distribution of
            attention spans.
    """

    offset = 0
    full_ranking = []
    n_items_to_rank = min(len(products), capacity)
    prods = copy(products)
    while n_items_to_rank > offset:
        x, ranking = best_x(products=prods, g=g,
                            capacity=n_items_to_rank - offset, offset=offset)
        if x == 0:
            break

        prods = {k: v for k, v in prods.items()
                 if k not in [name for name, _ in ranking]}
        offset += x
        full_ranking += ranking

    return full_ranking

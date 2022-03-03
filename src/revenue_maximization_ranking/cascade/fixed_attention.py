"""Optimal rankings for the fixed attention problem.

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

Functions
---------
    optimal_rankings:
        Optimal rankings for the fixed attention span problem.
"""

from copy import copy
from typing import Dict, Tuple
from collections import defaultdict

__all__ = ["optimal_rankings"]


def key(product_: Tuple[str, Dict]) -> Tuple[float, float]:
    """Lemma 1 sorting key for the fixed attention problem.

    An optimal ranking for the fixed attention problem is sorted
    by this key function.

    Parameters
    ----------
        product_: tuple[str, dict]
            A tuple describing a product, the first element is its name
            or id, the second element is a dictionary describing its
            revenue if purchased and its probability of being
            purchased.

    Returns
    -------
        tuple:
            A tuple with the revenue of the product as first element
            and its probability as the second element.

    """

    return product_[1]["revenue"], product_[1]["probability"]


def optimal_rankings(products: Dict, capacity: int) -> Tuple[Dict, Dict]:
    """Optimal rankings for the fixed attention span problem.

    Given a set of products this function calculates a solution for
    every possible fixed attention span problem with attention ranging
    from 1 up to a defined capacity.

    This implementation is inspired on "Algorithm 1" from
    https://arxiv.org/abs/2012.03800, but returns all of H[0, k]
    elements to avoid the need for the "AssortOpt" algorithm.

    Parameters
    ----------
        products: dict
            Dictionary with all the products, keys must be the products
            ids and values must be dictionaries with the revenue and
            probability of each product.
        capacity: int
            Maximum capacity of items to be displayed by the retailer.

    Returns
    -------
    rankings, revenues: Tuple[dict, dict]
        Optimal rankings and its corresponding revenues. Each possible
        attention span is the key on those dictionaries.
    """

    h = defaultdict(lambda: 0.0)
    assort = defaultdict(lambda: [])

    for k in range(1, min(len(products), capacity) + 1):
        for j, product in reversed(list(enumerate(sorted(products.items(),
                                                         key=key,
                                                         reverse=True)))):
            default = h[j + 1, k]
            alternative = (h[j + 1, k - 1]
                           + product[1]["probability"] * (product[1]["revenue"]
                                                          - h[j + 1, k - 1]))
            if alternative >= default:
                h[j, k] = alternative
                aux_list = copy(assort[j + 1, k - 1])
                aux_list.append(product)
                assort[j, k] = aux_list
            else:
                h[j, k] = default
                assort[j, k] = assort[j + 1, k]

    rankings = {k: sorted(assort[0, k], key=key, reverse=True)
                for k in range(1, min(len(products), capacity) + 1)}
    revenues = {k: h[0, k] for k in range(1, min(len(products), capacity) + 1)}

    return rankings, revenues

"""Cascade module.

Within this module we assume that the cascade model holds.

Cascade model:
  Let's say p_i is the probability of a customer purchasing product i
in a given search once he sees it. The cascade model assumes that the
user sees products sequentially and stops once he chooses a product.
With this model then actual purchase probabilities for products
displayed in order {1, 2, 3, ...} are: p_1,
                                       (1 - p_1) * p_2,
                                       (1 - p_1) * (1 - p_2) * p_3,
                                       ...

Attention:
  Any given user may have a determined attention for a search, that is
a maximum number of products that he is willing to look before giving
up and leaving the store empty-handed. Among all users the attention
span may be fixed or random.

Functions
---------
    full_best_x:
        Implements the full_best_x ranking on a dataframe.
    expected_revenue:
        It calculates the expected revenue for the cascade model.
"""

from revenue_maximization_ranking.cascade.dataframe import full_best_x, \
                                                           expected_revenue

__all__ = ["full_best_x", "expected_revenue"]

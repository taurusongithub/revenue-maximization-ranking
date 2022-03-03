"""Library types.

Types
-----
    DistributionLike:
        Any object of this type must implement the sf and pmf methods
        like a scipy.stats distribution.
"""

from typing import Union, Any
from scipy.stats._distn_infrastructure import rv_frozen

# A distribution like object for this library must implement the sf and
# pmf methods.
DistributionLike = Union[rv_frozen, Any]

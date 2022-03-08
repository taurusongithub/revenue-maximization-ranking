"""Revenue maximization on dataframes.

Dataframes are a common way to store information about products and
searches, in this module we provide some functions to interact with
pandas dataframes.

Functions
---------
    full_best_x:
        Implements the full_best_x ranking on a dataframe.
    expected_revenue:
        It calculates the expected revenue for the cascade model.
"""

import pandas as pd
from typing import Dict, List, Union, Tuple
from revenue_maximization_ranking.cascade.best_x import best_x_full_capacity
from revenue_maximization_ranking.cascade\
                                 .revenue import expected_revenue as exp_rev
from revenue_maximization_ranking._types import DistributionLike

__all__ = ["full_best_x", "expected_revenue"]


def load_dataframe(df: pd.DataFrame, revenue_col: str,
                   probability_col: str) -> Dict:
    """It loads data of products from a dataframe.

    Parameters
    ----------
        df: pandas.DataFrame
            The dataframe with the products' data.
        revenue_col: str
            Name of the column with the revenue for each product.
        probability_col: str
            Name of the column with the conditional probabilities
            of purchasing the product once it is seen by the user.

    Returns
    -------
        products: dict
            A dictionary with the products' info, keys are the index of
            the dataframe, values are dictionaries with the revenue and
            probability of the product as required by other functions
            in revenue_maximization_ranking.cascade.
    """

    df_ = df[[revenue_col, probability_col]].copy()
    products = df_.rename(columns={revenue_col: "revenue",
                                   probability_col: "probability"})\
                  .transpose().to_dict()
    return products


def ranking_as_column(ranking: List) -> pd.Series:
    """It converts a ranking list in a pandas.Series.

    Parameters
    ----------
        ranking: List
            A list of tuples representing products, the first element
            of each tuple will be the product id and the second element
            must be a dictionary with the revenue and probability of
            the product.

    Returns
    -------
        rank_column: pandas.Series
            A pandas.Series with product_id as index and the
            corresponding ranking (1, 2, 3, ...) of each product.
    """

    index = [name for name, _ in ranking]
    rank_column = pd.Series([i + 1 for i in range(len(index))], index=index)
    return rank_column


def full_best_x(df: pd.DataFrame, revenue_col: str, probability_col: str,
                g: DistributionLike, capacity: int = 0,
                show_xs: bool = False) -> Union[pd.Series,
                                                Tuple[pd.Series, List]]:
    """Implements the full_best_x ranking on a dataframe.

    Parameters
    ----------
        df: pandas.DataFrame
            Dataframe storing the products' data.
        revenue_col: str
            Name of the column with the revenue of each product.
        probability_col: str
            Name of the column with the conditional probability of each
            product.
        g: DistributionLike
            Distribution of attention spans.
        capacity: int, default 0
            Maximum number of products that the retailer can display.
        show_xs: bool, default False
            List of xs' values chosen by the algorithm.

    Returns
    -------
        rank_column: pandas.Series
            A pandas series with the ranking of each product. The index
            will be the same as the df index.
        best_xs: list, optional
            List of xs' values chosen by the algorithm.
    """

    if capacity < 1:
        capacity = df.shape[0]

    products = load_dataframe(df, revenue_col, probability_col)
    algorithm = best_x_full_capacity(products, g, capacity, show_xs=show_xs)
    if show_xs:
        rank_column = ranking_as_column(algorithm[0])
        best_xs = algorithm[1]
        return rank_column, best_xs

    return ranking_as_column(algorithm)


def expected_revenue(df: pd.DataFrame, revenue_col: str, probability_col: str,
                     ranking_col: str, g: DistributionLike) -> float:
    """It calculates the expected revenue for the cascade model.

    Parameters
    ----------
        df: pandas.DataFrame
            Dataframe storing the products' data.
        revenue_col: str
            Name of the column with the revenue of each product.
        probability_col: str
            Name of the column with the conditional probability of each
            product.
        g: DistributionLike
            Distribution of attention spans.
        ranking_col: str
            Name of the columns with the ranking to be evaluated.

    Returns
    -------
        revenue: float
            The expected revenue of the ranking according to the
            cascade model.
    """

    df_ = df[[revenue_col, probability_col, ranking_col]].copy()
    df_.query(f"{ranking_col}.notna()", inplace=True)
    df_.rename(columns={revenue_col: "revenue",
                        probability_col: "probability"}, inplace=True)
    df_.sort_values(by=ranking_col, inplace=True)
    df_.drop(columns=ranking_col, inplace=True)
    revenue = exp_rev(ranked_products=df_.iterrows(), g=g)
    return revenue

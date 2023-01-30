"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.4
"""

import pandas
import modin.pandas as pd
import torch
from torch_geometric.data import HeteroData
from typing import Tuple


def create_dataframe_graph_edges(
    movies: pandas.DataFrame,
    ratings: pandas.DataFrame,
) -> pd.DataFrame:
    """
    Creates new columns node_movie_id and node_user_id going to 0 up to the number or users or movies, to
    label the nodes in the graph.
    The output dataframe contains all edges between
    :param movies:
    :param ratings:
    :return:
    """

    # pandas to modin conversion
    movies = pd.DataFrame(movies)
    ratings = pd.DataFrame(ratings)

    movie_id_to_node_id = {
        movie_id: node_id for node_id, movie_id in enumerate(movies["movieId"].unique())
    }
    movies["movie_node_id"] = movies["movieId"].map(lambda x: movie_id_to_node_id[x])

    user_id_to_node_id = {
        user_id: node_id for node_id, user_id in enumerate(ratings["userId"].unique())
    }
    ratings["user_node_id"] = ratings["userId"].map(lambda x: user_id_to_node_id[x])

    edges_dataframe = pd.merge(ratings, movies, on="movieId", how="inner")

    return edges_dataframe[
        ["userId", "movieId", "rating", "title", "user_node_id", "movie_node_id"]
    ]


def create_dataframe_users_nodes_attributes(
    edges_dataframe: pd.DataFrame,
    tags: pandas.DataFrame,
) -> pd.DataFrame:
    tags = pd.DataFrame(tags)
    users_attributes_dataframe = pd.merge(edges_dataframe, tags, on="userId", how="inner")
    return users_attributes_dataframe


def create_dataframe_movies_nodes_attributes(
    edges_dataframe: pd.DataFrame,
    movies: pandas.DataFrame,
    genome_scores: pandas.DataFrame,
    genome_tags: pandas.DataFrame,
) -> pd.DataFrame:
    movies = pd.DataFrame(movies)
    genome_scores = pd.DataFrame(genome_scores)
    genome_tags = pd.DataFrame(genome_tags)

    genome_scores = pd.merge(
        genome_scores, edges_dataframe[["movieId", "movie_node_id"]], on="movieId", how="inner"
    )
    movies_tags_relevance = genome_scores.pivot(
        index="movie_node_id", columns="tagId", values="relevance"
    )

    genome_named_scores = pd.merge(genome_scores, genome_tags, on="tagId", how="inner")
    movies_attributes_dataframe = pd.merge(genome_named_scores, movies, on="movieId", how="inner")
    return movies_attributes_dataframe


def create_edges_tensors(edges_dataframe: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # source nodes
    src = edges_dataframe["user_node_id"].values.tolist()
    # destination nodes
    dst = edges_dataframe["movie_node_id"].values.tolist()
    # weight of each edge
    weight = edges_dataframe["rating"].values.tolist()

    edges_index = torch.tensor([src, dst], dtype=torch.long)
    edges_attr = torch.tensor(weight, dtype=torch.float)

    return edges_index, edges_attr


def create_nodes_tensors(
    users_attributes_dataframe: pd.DataFrame,
    movies_attributes_dataframe: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    movies_nodes_attr, users_nodes_attr = None, None
    return movies_nodes_attr, users_nodes_attr


def create_pyg_network(
    edges_index: torch.Tensor,
    edges_attr: torch.Tensor,
    movies_nodes_attr: torch.Tensor,
    users_nodes_attr: torch.Tensor,
) -> HeteroData:
    return HeteroData()

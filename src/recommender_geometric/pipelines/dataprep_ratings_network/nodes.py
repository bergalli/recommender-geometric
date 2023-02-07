"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""


from typing import Tuple

import numpy as np
import pandas
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def create_dataframe_graph_edges(
    movies: pandas.DataFrame,
    ratings: pandas.DataFrame,
) -> pd.DataFrame:
    """
    Creates new columns node_movie_id and node_user_id going to 0 up to the number or users or movies, to
    label the nodes in the graph.
    The output dataframe contains all edges of users ratings to movies
    :param movies:
    :param ratings:
    :return:
    """

    # pandas to modin conversion
    movies = pd.DataFrame(movies)
    ratings = pd.DataFrame(ratings)

    edges_dataframe = pd.merge(ratings, movies, on="movieId", how="inner")

    # assign nodes ids as a range to movieId and userId
    movie_id_to_node_id = {
        movie_id: node_id for node_id, movie_id in enumerate(edges_dataframe["movieId"].unique())
    }
    edges_dataframe["movie_node_id"] = edges_dataframe["movieId"].map(
        lambda x: movie_id_to_node_id[x]
    )
    user_id_to_node_id = {
        user_id: node_id for node_id, user_id in enumerate(edges_dataframe["userId"].unique())
    }
    edges_dataframe["user_node_id"] = edges_dataframe["userId"].map(lambda x: user_id_to_node_id[x])

    # convert ratings from a float [0,5] scale to an integer [0,100] scale
    edges_dataframe["rating"] = (
        (edges_dataframe["rating"] / edges_dataframe["rating"].max()) * 100
    ).astype(int)

    return edges_dataframe[
        ["userId", "movieId", "rating", "title", "user_node_id", "movie_node_id"]
    ]


# def create_dataframe_users_nodes_attributes(
#     edges_dataframe: pd.DataFrame,
#     tags: pandas.DataFrame,
# ) -> pd.DataFrame:
#     tags = pd.DataFrame(tags)
#     users_attributes_dataframe = pd.merge(
#         edges_dataframe, tags, on=["userId", "movieId"], how="inner"
#     )
#     return users_attributes_dataframe


# def create_dataframe_movies_nodes_attributes(
#     edges_dataframe: pd.DataFrame,
#     movies: pandas.DataFrame,
#     genome_scores: pandas.DataFrame,
#     genome_tags: pandas.DataFrame,
# ) -> pd.DataFrame:
#     movies = pd.DataFrame(movies)
#     genome_scores = pd.DataFrame(genome_scores)
#     genome_tags = pd.DataFrame(genome_tags)
#
#     movies_attributes = pd.merge(genome_scores, genome_tags, on="tagId", how="inner")
#     movies_attributes = pd.merge(
#         movies_attributes, edges_dataframe[["movieId", "movie_node_id"]], on="movieId", how="inner"
#     )
#     movies_attributes = movies_attributes.rename(columns={"relevance": "tag_relevance"})
#     movies_attributes = pd.merge(movies_attributes, movies, on="movieId", how="inner")
#
#     return movies_attributes[
#         ["movie_node_id", "movieId", "tagId", "tag_relevance", "tag", "title", "genres"]
#     ]


def define_users_to_movies_edges(
    edges_dataframe: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # source nodes
    src = edges_dataframe["user_node_id"].values.tolist()
    # destination nodes
    dst = edges_dataframe["movie_node_id"].values.tolist()
    # weight of each edge
    weight = edges_dataframe["rating"].values.tolist()

    edge_index = torch.tensor([src, dst])
    edge_label = torch.tensor(weight)

    return edge_index, edge_label


def define_movies_attributes(
    edges_dataframe: pd.DataFrame,
    genome_scores: pd.DataFrame,
) -> torch.Tensor:

    # merge node ids to movies genome scores
    # right join to keep all nodes in the output dataframe
    movie_id_to_node_id = edges_dataframe[["movieId", "movie_node_id"]].drop_duplicates()
    nodes_genome_scores = pd.merge(
        genome_scores,
        movie_id_to_node_id,
        on="movieId",
        how="right",
    )

    # pivot chunks with genome as columns and movie node id as index
    n_movies_per_chunk, unique_movies = 100, nodes_genome_scores["movie_node_id"].unique()
    pivoted_chunks = []
    for movies_chunk in np.array_split(unique_movies, n_movies_per_chunk):
        chunk = nodes_genome_scores.loc[nodes_genome_scores["movie_node_id"].isin(movies_chunk), :]
        pivoted = chunk.pivot(index="movie_node_id", columns="tagId", values="relevance")
        pivoted_chunks.append(pivoted)
    pivoted_nodes_genome_scores = pd.concat(pivoted_chunks, axis=0)
    pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.sort_index()

    # fill missing genome relevances with 0, for movies without a defined genome
    pivoted_nodes_genome_scores = pivoted_nodes_genome_scores.fillna(0)

    # convert movies attributes dataframe to torch tensor
    # todo move outside of func to log the dataframe in kedro
    movies_nodes_attr = pivoted_nodes_genome_scores.values.tolist()
    movies_nodes_attr = torch.tensor(movies_nodes_attr)

    return movies_nodes_attr


def create_pyg_network(
    edge_index: torch.Tensor,
    edge_label: torch.Tensor,
    movies_nodes_attr: torch.Tensor,
    # users_nodes_attr: torch.Tensor,
) -> HeteroData:

    data = HeteroData()

    data["movie"].x = movies_nodes_attr
    n_users = len(set(edge_index[0, :].tolist()))
    data["user"].x = torch.eye(n_users)

    data["user", "rates", "movie"].edge_index = edge_index
    data["user", "rates", "movie"].edge_label = edge_label

    # data_list = []
    # batch_size=1000
    # for idx in np.arange(0, edge_index.shape[1], batch_size):
    #     train_data = HeteroData()
    #
    #     train_data["movie"].x = movies_nodes_attr
    #     n_users = len(set(edge_index[0, :].tolist()))
    #     train_data["user"].x = torch.eye(n_users)
    #
    #     train_data["user", "rates", "movie"].edge_index = edge_index[:, idx:idx + batch_size]
    #     train_data["user", "rates", "movie"].edge_label = edge_label[idx:idx + batch_size]
    #
    #     # Add a reverse ('movie', 'rev_rates', 'user') relation for message passing.
    #     train_data = ToUndirected()(train_data)
    #     del train_data["movie", "rev_rates", "user"].edge_label  # Remove "reverse" label.
    #     data_list.append(train_data)

    return data

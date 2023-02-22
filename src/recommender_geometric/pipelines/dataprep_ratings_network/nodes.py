"""
This is a boilerplate pipeline 'make_ratings_network'
generated using Kedro 0.18.4
"""

from typing import Tuple

from pyspark.context import SparkContext
from pyspark.sql import DataFrame, Row, Window
from pyspark.sql.functions import col, create_map, lit, broadcast, last
import pyspark.sql.functions as fn
from pyspark.sql.types import IntegerType

from itertools import chain


def create_dataframe_graph_edges(
    ratings: DataFrame,
) -> Tuple[DataFrame, dict, dict]:
    """
    Creates new columns node_movie_id and node_user_id going to 0 up to the number or users or movies, to
    label the nodes in the graph.
    The output dataframe contains all edges of users ratings to movies
    :param movies:
    :param ratings:
    :return:
    """

    # assign nodes ids as a range to movieId
    unique_movie_ids = ratings.select("movieId").distinct().rdd.map(lambda r: r[0]).collect()
    movie_id_to_node_id = {movie_id: node_id for node_id, movie_id in enumerate(unique_movie_ids)}
    mapping_expr = create_map([lit(x) for x in chain(*movie_id_to_node_id.items())])
    ratings = ratings.withColumn("movie_node_id", mapping_expr[col("movieId")])

    # assign nodes ids as a range to userId
    unique_user_ids = ratings.select("userId").distinct().rdd.map(lambda r: r[0]).collect()
    user_id_to_node_id = {user_id: node_id for node_id, user_id in enumerate(unique_user_ids)}
    mapping_expr = create_map([lit(x) for x in chain(*user_id_to_node_id.items())])
    ratings = ratings.withColumn("user_node_id", mapping_expr[col("userId")])

    # convert ratings from a float [0,5] scale to an integer [0,100] scale
    ratings = (
        ratings.crossJoin(ratings.select(fn.max("rating").alias("max_rating")))
        .withColumn("rating", (col("rating") / col("max_rating")) * 100)
        .drop("max_rating")
        .withColumn("rating", col("rating").cast(IntegerType()))
    )

    edges_dataframe = ratings
    return edges_dataframe, movie_id_to_node_id, user_id_to_node_id


def define_movies_attributes(
    movie_id_to_node_id,
    genome_scores: DataFrame,
) -> DataFrame:

    movie_id_to_node_id_mapper = create_map([lit(x) for x in chain(*movie_id_to_node_id.items())])
    genome_scores = genome_scores.withColumn("movie_node_id", movie_id_to_node_id_mapper[col("movieId")])

    # fill in missing values for movies without a genome
    sc = SparkContext.getOrCreate()
    all_movie_node_id = sc.range(min(movie_id_to_node_id.values()),
                                 max(movie_id_to_node_id.values())).map(Row("movie_node_id")).toDF()
    all_movie_node_id = all_movie_node_id.crossJoin(genome_scores.select("tagId")).cache()
    genome_scores = genome_scores.join(broadcast(all_movie_node_id), on=["movie_node_id", "tagId"], how="outer") \
        .withColumn(
        "relevance",
        last(
            "relevance",
            ignorenulls=True
        ).over(
            Window.partitionBy("tagId").orderBy("movie_node_id") \
                .rowsBetween(Window.unboundedPreceding, 0)
        )
    )

    pivoted_genome_scores = genome_scores.groupBy("movie_node_id").pivot("tagId").max("relevance")
    pivoted_genome_scores = pivoted_genome_scores.orderBy(col("movie_node_id"))
    pivoted_genome_scores = pivoted_genome_scores.drop("movie_node_id")
    # sort index so that rows are ordered according to movie node id
    # pivoted_genome_scores["integer_index"] = pivoted_genome_scores.index.astype(int)
    # pivoted_genome_scores = pivoted_genome_scores.set_index(
    #     keys="integer_index", drop=True
    # )
    # pivoted_genome_scores = pivoted_genome_scores.sort_index()
    #
    # # fill missing genome relevances with 0, for movies without a defined genome
    # pivoted_genome_scores = pivoted_genome_scores.fillna(0)

    return pivoted_genome_scores


def define_users_to_movies_edges(
    edges_dataframe: DataFrame,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Keep single column dataframes because kedro not handling pandas spark series
    """
    # source nodes
    src = edges_dataframe.select(col("user_node_id"))
    # destination nodes
    dst = edges_dataframe.select(col("movie_node_id"))
    # weight of each edge
    weight = edges_dataframe.select(col("rating"))

    return src, dst, weight

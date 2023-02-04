"""
This is a boilerplate pipeline 'recommender_model'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_gcn_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                train_gcn_model,
                inputs=dict(
                    train_data="train_data",
                    val_data="val_data",
                    test_data="test_data",
                ),
                outputs="trained_model",
                name="train_gcn_model",
            )
        ]
    )

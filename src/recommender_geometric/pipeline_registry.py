"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline, pipeline
import os

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    backend = os.environ["RECOMMENDER_GEOMETRIC_BACKEND"]

    pipelines = find_pipelines()
    pipelines["__default__"] = pipeline(sum(pipelines.values()), namespace=backend)
    return pipelines

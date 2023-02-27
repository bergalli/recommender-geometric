from torch_geometric.datasets import OGB_MAG
from torch_geometric.data.feature_store import FeatureStore, FeatureTensorType, TensorAttr
from typing import Optional
from torch_geometric.data.graph_store import GraphStore
from petastorm.spark import make_spark_converter
import torch_geometric.transforms as T
import os.path as osp

obg = OGB_MAG(osp.join('data', 'OGB_MAG'), preprocess='metapath2vec', transform=T.ToUndirected(merge=False))
data = obg[0]


class SparkFeatureStore(FeatureStore):
    def _put_tensor(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        pass

    def _get_tensor(self, attr: TensorAttr) -> Optional[FeatureTensorType]:
        pass

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        pass


fs = SparkFeatureStore()
fs["paper", "x", None] = data["paper"].x
from typing import NamedTuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer

LayerMapping = NamedTuple("LayerMapping", [("from_layer", str), ("to_layer", str)])


def transfert_weights(from_model: Model, to_model: Model, layer_mapping: list[LayerMapping]):
    for mapping in layer_mapping:
        from_layer: Layer = from_model.get_layer(mapping.from_layer)
        to_layer: Layer = to_model.get_layer(mapping.to_layer)
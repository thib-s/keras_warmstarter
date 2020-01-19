from typing import NamedTuple
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Conv2D

from warmstarter.strategies import warmstart_Dense, warmstart_Conv2D

LayerMapping = NamedTuple("LayerMapping", [("from_layer", str), ("to_layer", str)])

STRATEGIES = {
    (Dense, Dense): warmstart_Dense,
    (Conv2D, Conv2D): warmstart_Conv2D
}


def transfert_weights(from_model: Model, to_model: Model, layer_mapping: list[LayerMapping]):
    for mapping in layer_mapping:
        from_layer: Layer = from_model.get_layer(mapping.from_layer)
        to_layer: Layer = to_model.get_layer(mapping.to_layer)
        STRATEGIES[(from_layer.__class__, to_layer.__class__)](from_layer, to_layer)
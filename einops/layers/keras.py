__author__ = "Alex Rogozhnikov"

from einops.layers.tensorflow import EinMix, Rearrange, Reduce

keras_custom_objects = {
    Rearrange.__name__: Rearrange,
    Reduce.__name__: Reduce,
    EinMix.__name__: EinMix,
}

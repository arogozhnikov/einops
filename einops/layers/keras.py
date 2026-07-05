__author__ = "Alex Rogozhnikov"

from einops.layers.tensorflow import EinMix, Rearrange, Reduce, Repeat

keras_custom_objects = {
    Rearrange.__name__: Rearrange,
    Reduce.__name__: Reduce,
    Repeat.__name__: Repeat,
    EinMix.__name__: EinMix,
}

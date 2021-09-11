__author__ = 'Alex Rogozhnikov'

from ..layers.tensorflow import Rearrange, Reduce, WeightedEinsum

keras_custom_objects = {
    Rearrange.__name__: Rearrange,
    Reduce.__name__: Reduce,
    WeightedEinsum.__name__: WeightedEinsum,
}

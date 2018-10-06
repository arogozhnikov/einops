__author__ = 'Alex Rogozhnikov'

from einops import transpose, reduce


# TODO tests for serialization / deserialization inside the model
# TODO parsing to recipe at the creation stage + think of serialization for the following history
# TODO reductions
# TODO documenation
# TODO make imports like from einops.torch import ...

def _repr_layer(layer):
    params = repr(layer.pattern)
    for axis, length in layer.axes_lengths.items():
        params += ', {}={}'.format(axis, length)
    return '{}({})'.format(layer.__class__.__name__, params)


class TorchSubmodule:
    def __getattr__(self, item):
        import torch

        class Transpose(torch.nn.Module):
            def __init__(self, pattern, **axes_lengths):
                super(Transpose, self).__init__()
                self.pattern = pattern
                self.axes_lengths = axes_lengths

            def forward(self, input):
                return transpose(tensor=input, pattern=self.pattern, **self.axes_lengths)

            __repr__ = _repr_layer

        self.Transpose = Transpose

        return self.__dict__[item]

    def __dir__(self):
        return ['Transpose', 'Reduce']


torch = TorchSubmodule()


class ChainerSubmodule:
    def __getattr__(self, item):
        import chainer

        class Transpose(chainer.Link):
            def __init__(self, pattern, **axes_lengths):
                super(Transpose, self).__init__()
                self.pattern = pattern
                self.axes_lengths = axes_lengths

            def __call__(self, x):
                return transpose(tensor=x, pattern=self.pattern, **self.axes_lengths)

            __repr__ = _repr_layer

        self.Transpose = Transpose

        return self.__dict__[item]

    def __dir__(self):
        return ['Transpose', 'Reduce']


chainer = ChainerSubmodule()


class GluonSubmodule:
    def __getattr__(self, item):
        import mxnet

        class Transpose(mxnet.gluon.Block):
            def __init__(self, pattern, **axes_lengths):
                super(Transpose, self).__init__()
                self.pattern = pattern
                self.axes_lengths = axes_lengths

            def forward(self, x):
                return transpose(tensor=x, pattern=self.pattern, **self.axes_lengths)

            __repr__ = _repr_layer

        self.Transpose = Transpose

        return self.__dict__[item]

    def __dir__(self):
        return ['Transpose', 'Reduce']


gluon = GluonSubmodule()

__author__ = 'Alex Rogozhnikov'

import layers

import torch
import chainer
import mxnet
import numpy
import backends


def test_transpose():
    backend_pairs = [
        (backends.TorchBackend(), layers.torch.Transpose),
        (backends.ChainerBackend(), layers.chainer.Transpose),
        (backends.MXNetNdarrayBackend(), layers.gluon.Transpose),
    ]
    patterns = [
        ('b c h w -> b (c h w)', dict(b=10), (10, 20 * 30 * 40)),
        ('b c (h1 h2) (w1 w2) -> b (c h2 w2) h1 w1', dict(h1=15, h2=2, w2=2), (10, 20 * 2 * 2, 15, 20)),
    ]

    for backend, TransposeLayer in backend_pairs:
        for pattern, axes_lengths, result_shape in patterns:
            x = numpy.arange(10 * 20 * 30 * 40, dtype='float32').reshape([10, 20, 30, 40])
            layer = TransposeLayer(pattern, **axes_lengths)
            assert layer(backend.from_numpy(x)).shape == result_shape
            for shape in [(), (10,), (10, 10, 10), (15, 20, 31, 40), (10, 1, 1, 1, 1)]:
                try:
                    layer(backend.from_numpy(numpy.zeros(shape, dtype='float32')))
                except:
                    pass
                else:
                    raise AssertionError('Failure expected')

        x = numpy.arange(10 * 20 * 30 * 40, dtype='float32').reshape([10, 20, 30, 40])
        x = backend.from_numpy(x)
        if isinstance(x, torch.Tensor):
            x.requires_grad = True
        if isinstance(x, mxnet.nd.NDArray):
            x.attach_grad()

        if isinstance(x, chainer.Variable):
            chainer.functions.sum(layer(x)).backward()
        elif isinstance(x, mxnet.nd.NDArray):
            from mxnet import autograd
            with autograd.record():
                layer(x).sum().backward()
        else:
            layer(x).sum().backward()

        assert numpy.allclose(backend.to_numpy(x.grad), 1)

        print('Tested layer for ', backend.framework_name)


test_transpose()

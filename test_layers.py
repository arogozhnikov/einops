__author__ = 'Alex Rogozhnikov'

import pickle

import layers

import torch
import chainer
import mxnet
import numpy
import backends


def test_transpose():
    backend_pairs = [
        (backends.TorchBackend(), layers.TorchTranspose),
        (backends.ChainerBackend(), layers.ChainerTranspose),
        (backends.MXNetNdarrayBackend(), layers.GluonTranspose),
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

            # simple pickling / unpickling
            layer2 = pickle.loads(pickle.dumps(layer))
            result1 = backend.to_numpy(layer(backend.from_numpy(x)))
            result2 = backend.to_numpy(layer2(backend.from_numpy(x)))
            assert numpy.allclose(result1, result2)

            v = backend.from_numpy(x)
            if isinstance(v, torch.Tensor):
                v.requires_grad = True
            if isinstance(v, mxnet.nd.NDArray):
                v.attach_grad()

            if isinstance(v, chainer.Variable):
                chainer.functions.sum(layer(v)).backward()
            elif isinstance(v, mxnet.nd.NDArray):
                from mxnet import autograd
                with autograd.record():
                    layer(v).sum().backward()
            else:
                layer(v).sum().backward()

            assert numpy.allclose(backend.to_numpy(v.grad), 1)

        print('Tested layer for ', backend.framework_name)


test_transpose()

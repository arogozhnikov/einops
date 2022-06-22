import os

from einops import _backends
import warnings

__author__ = 'Alex Rogozhnikov'

import logging

# minimize noise in tests logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True

flag_to_bool = {
    '': False,
    '0': False,
    '1': True,
}

skip_cupy = flag_to_bool[os.environ.get('EINOPS_SKIP_CUPY', '1')]
skip_oneflow = flag_to_bool[os.environ.get('EINOPS_SKIP_ONEFLOW', '1')]
skip_mindspore = flag_to_bool[os.environ.get('EINOPS_SKIP_MINDSPORE', '1')]

def collect_test_backends(symbolic=False, layers=False):
    """
    :param symbolic: symbolic or imperative frameworks?
    :param layers: layers or operations?
    :return: list of backends satisfying set conditions
    """
    if not symbolic:
        if not layers:
            backend_types = [
                _backends.NumpyBackend,
                _backends.JaxBackend,
                _backends.TorchBackend,
                _backends.GluonBackend,
                _backends.ChainerBackend,
                _backends.TensorflowBackend,
                _backends.OneFlowBackend,
                _backends.MindSporeBackend,

            ]
            if not skip_cupy:
                backend_types += [_backends.CupyBackend]
        else:
            backend_types = [
                _backends.TorchBackend,
                _backends.GluonBackend,
                _backends.ChainerBackend,
                _backends.OneFlowBackend,
                _backends.MindSporeBackend,
            ]
    else:
        if not layers:
            backend_types = [
                _backends.MXNetBackend,
            ]
        else:
            backend_types = [
                _backends.MXNetBackend,
                _backends.KerasBackend,
            ]

    result = []
    for backend_type in backend_types:
        try:
            result.append(backend_type())
        except ImportError:
            # problem with backend installation fails a specific test function,
            # but will be skipped in all other test cases
            warnings.warn('backend could not be initialized for tests: {}'.format(backend_type))
    return result

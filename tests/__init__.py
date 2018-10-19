import os

import tensorflow as tf

from einops import backends

__author__ = 'Alex Rogozhnikov'


assert os.environ.get('TF_EAGER', '') in ['', '1', '0']
if os.environ.get('TF_EAGER', '') == '1':
    tf.enable_eager_execution()
    print('testing with eager execution')


def collect_test_backends(symbolic=False, layers=False):
    """
    :param symbolic: symbolic or imperative frameworks?
    :param layers: layers or operation?
    :return: list of backends satisfying set conditions
    """
    tf_running_eagerly = True
    try:
        import tensorflow
        tf_running_eagerly = tensorflow.executing_eagerly()
    except ImportError:
        print("Couldn't import tensorflow for testing")

    if not symbolic:
        if not layers:
            backend_types = [backends.NumpyBackend,
                             backends.CupyBackend,
                             backends.TorchBackend,
                             backends.GluonBackend,
                             backends.ChainerBackend,
                             ]
            if tf_running_eagerly:
                backend_types += [backends.TensorflowBackend]
        else:
            backend_types = [backends.TorchBackend,
                             backends.GluonBackend,
                             backends.ChainerBackend]
    else:
        if not layers:
            backend_types = [backends.MXNetBackend]
            if not tf_running_eagerly:
                backend_types += [backends.TensorflowBackend]
        else:
            backend_types = [backends.MXNetBackend]
            if not tf_running_eagerly:
                backend_types += [backends.KerasBackend]

    result = []
    for backend_type in backend_types:
        try:
            result.append(backend_type())
        except ImportError:
            pass
    return result

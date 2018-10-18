import os

import tensorflow as tf

from einops import backends

__author__ = 'Alex Rogozhnikov'


if bool(os.environ.get('TF_EAGER', False)):
    tf.enable_eager_execution()
    print('testing with eager execution')


def collect_test_settings():
    # TODO add only when available
    tf_running_eagerly = True
    try:
        import tensorflow
        tf_running_eagerly = tensorflow.executing_eagerly()
    except ImportError:
        print("Couldn't import tensorflow for testing")
    testing_settings = {}
    for backend_type in [backends.NumpyBackend,
                         backends.CupyBackend,
                         backends.TorchBackend,
                         backends.GluonBackend,
                         backends.ChainerBackend,
                         ] + ([backends.TensorflowBackend] if tf_running_eagerly else []):
        try:
            backend = backend_type()
            testing_settings[backend.framework_name, 'imperative', 'operation'] = dict(
                backend=backend
            )
        except ImportError:
            pass

    for backend_type in [backends.TorchBackend,
                         backends.GluonBackend,
                         backends.ChainerBackend]:
        try:
            backend = backend_type()
            testing_settings[backend.framework_name, 'imperative', 'layer'] = dict(
                backend=backend,
                layers=backend.layers(),
                savers_loaders=[],  # TODO
            )
        except ImportError:
            pass

    for backend_type in [backends.MXNetBackend] + ([] if tf_running_eagerly else [backends.TensorflowBackend]):
        try:
            backend = backend_type()
            testing_settings[backend.framework_name, 'symbolic', 'operation'] = dict(
                backend=backend
            )
        except ImportError:
            pass

    for backend_type in [backends.KerasBackend, backends.MXNetBackend]:
        try:
            backend = backend_type()
            testing_settings[backend.framework_name, 'symbolic', 'layer'] = dict(
                backend=backend,
                layers=backend.layers(),
                savers_loaders=[],  # TODO
            )
        except ImportError:
            pass

    return testing_settings

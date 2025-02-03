"""
Common utils for testing.
These functions allow testing only some frameworks, not all.
"""

import logging
import os
from functools import lru_cache
from typing import List, Tuple

from einops import _backends
import warnings

__author__ = "Alex Rogozhnikov"


# minimize noise in tests logging
logging.getLogger("tensorflow").disabled = True
logging.getLogger("matplotlib").disabled = True

FLOAT_REDUCTIONS = ("min", "max", "sum", "mean", "prod")  # not includes any/all


def find_names_of_all_frameworks() -> List[str]:
    backend_subclasses = []
    backends = _backends.AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)
    return [b.framework_name for b in backend_subclasses]


ENVVAR_NAME = "EINOPS_TEST_BACKENDS"


def unparse_backends(backend_names: List[str]) -> Tuple[str, str]:
    _known_backends = find_names_of_all_frameworks()
    for backend_name in backend_names:
        if backend_name not in _known_backends:
            raise RuntimeError(f"Unknown framework: {backend_name}")
    return ENVVAR_NAME, ",".join(backend_names)


@lru_cache(maxsize=1)
def parse_backends_to_test() -> List[str]:
    if ENVVAR_NAME not in os.environ:
        raise RuntimeError(f"Testing frameworks were not specified, env var {ENVVAR_NAME} not set")
    parsed_backends = os.environ[ENVVAR_NAME].split(",")
    _known_backends = find_names_of_all_frameworks()
    for backend_name in parsed_backends:
        if backend_name not in _known_backends:
            raise RuntimeError(f"Unknown framework: {backend_name}")

    return parsed_backends


def is_backend_tested(backend: str) -> bool:
    """Used to skip test if corresponding backend is not tested"""
    if backend not in find_names_of_all_frameworks():
        raise RuntimeError(f"Unknown framework {backend}")
    return backend in parse_backends_to_test()


def collect_test_backends(symbolic=False, layers=False) -> List[_backends.AbstractBackend]:
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
                _backends.TensorflowBackend,
                _backends.OneFlowBackend,
                _backends.PaddleBackend,
                _backends.CupyBackend,
            ]
        else:
            backend_types = [
                _backends.TorchBackend,
                _backends.OneFlowBackend,
                _backends.PaddleBackend,
            ]
    else:
        if not layers:
            backend_types = [
                _backends.PyTensorBackend,
            ]
        else:
            backend_types = [
                _backends.TFKerasBackend,
            ]

    backend_names_to_test = parse_backends_to_test()
    result = []
    for backend_type in backend_types:
        if backend_type.framework_name not in backend_names_to_test:
            continue
        try:
            result.append(backend_type())
        except ImportError:
            # problem with backend installation fails a specific test function,
            # but will be skipped in all other test cases
            warnings.warn("backend could not be initialized for tests: {}".format(backend_type))
    return result

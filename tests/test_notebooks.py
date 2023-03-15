from typing import Dict

from io import StringIO

from tests import collect_test_backends

__author__ = "Alex Rogozhnikov"

from pathlib import Path
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


def render_notebook(filename: Path, replacements: Dict[str, str]) -> str:
    """Takes path to the notebook, returns executed and rendered version
    :param filename: notebook
    :param replacements: dictionary with text replacements done before executing
    :return: notebook, rendered as string
    """
    with filename.open("r") as f:
        nb_as_str = f.read()
    for original, replacement in replacements.items():
        nb_as_str = nb_as_str.replace(original, replacement)

    nb = nbformat.read(StringIO(nb_as_str), nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=60, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(filename.parent.absolute())}})

    result_as_stream = StringIO()
    nbformat.write(nb, result_as_stream)
    return result_as_stream.getvalue()


def test_notebook_1():
    [notebook] = Path(__file__).parent.with_name("docs").glob("1-*.ipynb")
    render_notebook(notebook, replacements={})


def test_notebook_2_with_all_backends():
    [notebook] = Path(__file__).parent.with_name("docs").glob("2-*.ipynb")
    backends = []
    if "chainer" in collect_test_backends(symbolic=False, layers=True):
        backends += ["chainer"]
    if "pytorch" in collect_test_backends(symbolic=False, layers=True):
        backends += ["pytorch"]
    if "tensorflow" in collect_test_backends(symbolic=False, layers=False):
        backends += ["tensorflow"]
    for backend in backends:
        print("Testing {} with backend {}".format(notebook, backend))
        replacements = {"flavour = 'pytorch'": "flavour = '{}'".format(backend)}
        expected_string = "selected {} backend".format(backend)
        result = render_notebook(notebook, replacements=replacements)
        assert expected_string in result


def test_notebook_3():
    [notebook] = Path(__file__).parent.with_name("docs").glob("3-*.ipynb")
    if "pytorch" not in collect_test_backends(symbolic=False, layers=True):
        pytest.skip()
    render_notebook(notebook, replacements={})


def test_notebook_4():
    [notebook] = Path(__file__).parent.with_name("docs").glob("4-*.ipynb")
    if "pytorch" not in collect_test_backends(symbolic=False, layers=True):
        pytest.skip()
    render_notebook(notebook, replacements={})

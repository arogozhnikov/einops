from typing import Dict

from io import StringIO

from tests import is_backend_tested

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
        assert original in nb_as_str, f"not found in notebook: {original}"
        nb_as_str = nb_as_str.replace(original, replacement)

    nb = nbformat.read(StringIO(nb_as_str), nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
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
    if is_backend_tested("torch"):
        # notebook uses name pytorch
        backends.append("pytorch")
    if is_backend_tested("tensorflow"):
        backends.append("tensorflow")
    if is_backend_tested("chainer"):
        backends.append("chainer")

    if len(backends) == 0:
        pytest.skip()

    for backend in backends:
        print("Testing {} with backend {}".format(notebook, backend))
        replacements = {"flavour = 'pytorch'": "flavour = '{}'".format(backend)}
        expected_string = "selected {} backend".format(backend)
        result = render_notebook(notebook, replacements=replacements)
        assert expected_string in result


def test_notebook_3():
    [notebook] = Path(__file__).parent.with_name("docs").glob("3-*.ipynb")
    if not is_backend_tested("torch"):
        pytest.skip()
    render_notebook(notebook, replacements={})


def test_notebook_4():
    [notebook] = Path(__file__).parent.with_name("docs").glob("4-*.ipynb")
    if not is_backend_tested("torch"):
        pytest.skip()
    render_notebook(notebook, replacements={})

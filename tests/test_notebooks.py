from typing import Dict

from io import StringIO

from tests import collect_test_backends

__author__ = 'Alex Rogozhnikov'

from pathlib import Path
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def render_notebook(filename: Path, replacements: Dict[str, str]) -> str:
    """ Takes path to the notebook, returns executed and rendered version
    :param filename: notebook
    :param replacements: dictionary with text replacements done before executing
    :return: notebook, rendered as string
    """
    with filename.open('r') as f:
        nb_as_str = f.read()
    for original, replacement in replacements.items():
        nb_as_str = nb_as_str.replace(original, replacement)

    nb = nbformat.read(StringIO(nb_as_str), nbformat.NO_CONVERT)
    ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': str(filename.parent.absolute())}})

    result_as_stream = StringIO()
    nbformat.write(nb, result_as_stream)
    return result_as_stream.getvalue()


def test_all_notebooks():
    notebooks = Path(__file__).parent.with_name('docs').glob('*.ipynb')
    for notebook in notebooks:
        render_notebook(notebook, replacements={})


def test_dl_notebook_with_all_backends():
    notebook, = Path(__file__).parent.with_name('docs').glob('2-*.ipynb')
    backends = ['chainer', 'pytorch']
    if 'tensorflow' in collect_test_backends(symbolic=False, layers=False):
        backends += ['tensorflow']
    for backend in backends:
        print('Testing {} with backend {}'.format(notebook, backend))
        replacements = {"flavour = 'pytorch'": "flavour = '{}'".format(backend)}
        expected_string = 'selected {} backend'.format(backend)
        result = render_notebook(notebook, replacements=replacements)
        assert expected_string in result

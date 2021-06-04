"""
Usage: python test.py
1. Installs part of dependencies (make sure `which pip` points to correct location)
2. Installs current version of einops in editable mode
3. Runs tests
"""

import os
from subprocess import Popen, PIPE
from pathlib import Path

__author__ = 'Alex Rogozhnikov'


def run(cmd, **env):
    # keeps printing output when testing
    cmd = cmd.split(' ') if isinstance(cmd, str) else cmd
    p = Popen(cmd, cwd=str(Path(__file__).parent), env={**os.environ, **env})
    p.communicate()
    return p.returncode


# check we have nvidia-smi
output, _ = Popen('which nvidia-smi'.split(' '), stdout=PIPE).communicate()
have_cuda = b'nvidia' in output

# install cupy. It can't be installed without cuda available (with compilers).
if have_cuda:
    return_code = run('pip install cupy --pre --progress-bar off')
    assert return_code == 0

# install dependencies
dependencies = [
    'numpy',
    'mxnet==1.*',
    'torch',
    'tensorflow',
    'chainer',
    'jax',
    'jaxlib',
    'nbformat',
    'nbconvert',
    'jupyter',
    'pillow',
    'nose',
]

assert 0 == run('pip install {} --progress-bar off'.format(' '.join(dependencies)))
# install einops
assert 0 == run('pip install -e .')


return_code = run('python -m nose tests -vds', EINOPS_SKIP_CUPY='0' if have_cuda else '1')
assert return_code == 0

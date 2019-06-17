"""
Usage: python test.py
1. Installs part of dependencies
2. Installs current version of einops in editable mode
3. Runs tests
"""

import sys
import os
from subprocess import Popen, PIPE
from pathlib import Path

__author__ = 'Alex Rogozhnikov'


def run(cmd, **env):
    # keeps printing output when testing
    cmd = cmd.split(' ') if isinstance(cmd, str) else cmd
    p = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr,
              cwd=str(Path(__file__).parent), env={**os.environ, **env})
    p.communicate()
    return p.returncode


# check we have nvidia-smi
output, _ = Popen('which nvidia-smi'.split(' '), stdout=PIPE).communicate()
have_cuda = b'nvidia' in output

# install cupy
if have_cuda:
    return_code = run('pip install cupy --pre --progress-bar off')
    assert return_code == 0

# install dependencies
dependencies = [
    'numpy',
    'mxnet',
    'torch',
    'tensorflow',
    'chainer',
    'keras',
    'nbformat',
    'nbconvert',
    'jupyter',
    'pillow',
    'nose',
]
assert 0 == run('pip install {} --pre --progress-bar off'.format(' '.join(dependencies)))
# install einops
assert 0 == run('pip install -e .')


# we need to run tests twice
# - once for tensorflow eager
return_code1 = run('nosetests tests -vds', TF_EAGER='1', EINOPS_SKIP_CUPY='0' if have_cuda else '1')
print('\n' * 5)
# - and once for symbolic tensorflow
return_code2 = run('nosetests tests -vds', TF_EAGER='0', EINOPS_SKIP_CUPY='0' if have_cuda else '1')

assert return_code1 == 0 and return_code2 == 0

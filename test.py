import sys
import os
from subprocess import Popen, PIPE
from pathlib import Path

__author__ = 'Alex Rogozhnikov'


def run(cmd, **env):
    # keeps printing output when testing
    cmd = cmd.split(' ') if isinstance(cmd, str) else cmd
    p = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, bufsize=10,
              universal_newlines=True, cwd=str(Path(__file__).parent), env={**os.environ, **env})
    p.communicate()
    return p.returncode


dependencies = [
    'numpy',
    'mxnet',
    'torch',
    'tensorflow',
    'chainer',
    'keras',
    'nbformat',
]

# check we have nvidia-smi
output, _ = Popen('which nvidia-smi'.split(' '), stdout=PIPE).communicate()
have_cuda = b'nvidia' in output

if have_cuda:
    return_code = run('pip install cupy --pre')
    assert return_code == 0

assert 0 == run('pip install {} --pre'.format(' '.join(dependencies)))

# we need to run tests twice
# -once for tensorflow eager
return_code1 = run('nosetests tests -vds', TF_EAGER='1', EINOPS_SKIP_CUPY='0' if have_cuda else '1')
print('\n' * 5)
# - and once for symbolic tensorflow
return_code2 = run('nosetests tests -vds', TF_EAGER='0', EINOPS_SKIP_CUPY='0' if have_cuda else '1')

assert return_code1 == 0 and return_code2 == 0

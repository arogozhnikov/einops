"""
Usage: python test.py <frameworks>

1. Installs part of dependencies (make sure `which pip` points to correct location)
2. Installs current version of einops in editable mode
3. Runs the tests
"""

import os
import sys
from subprocess import Popen
from pathlib import Path

__author__ = "Alex Rogozhnikov"


def run(cmd, **env):
    # keeps printing output when testing
    cmd = cmd.split(" ") if isinstance(cmd, str) else cmd
    p = Popen(cmd, cwd=str(Path(__file__).parent), env={**os.environ, **env})
    p.communicate()
    return p.returncode


def main():
    _executable, *frameworks = sys.argv
    framework_name2installation = {
        "numpy": ["numpy"],
        "torch": ["torch --index-url https://download.pytorch.org/whl/cpu"],
        "jax": ["jax[cpu]", "jaxlib", "flax"],
        "tensorflow": ["tensorflow"],
        "cupy": ["cupy"],
        # switch to stable paddlepaddle, because of https://github.com/PaddlePaddle/Paddle/issues/63927
        # "paddle": ["paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html"],
        "paddle": ["paddlepaddle"],
        "oneflow": ["oneflow==0.9.0"],
    }

    usage = f"""
    Usage:   python test.py <frameworks>
    Example: python test.py numpy pytorch

    Available frameworks: {list(framework_name2installation)}
    """
    if len(frameworks) == 0:
        print(usage)
        return
    else:
        synonyms = {
            "tf": "tensorflow",
            "pytorch": "torch",
            "paddlepaddle": "paddle",
        }
        frameworks = [synonyms.get(f, f) for f in frameworks]
        wrong_frameworks = [f for f in frameworks if f not in framework_name2installation]
        if wrong_frameworks:
            print(usage)
            raise RuntimeError(f"Unrecognized frameworks: {wrong_frameworks}")

    other_dependencies = [
        "nbformat",
        "nbconvert",
        "jupyter",
        "parameterized",
        "pillow",
        "pytest",
    ]
    for framework in frameworks:
        print(f"Installing {framework}")
        pip_instructions = framework_name2installation[framework]
        assert 0 == run("pip install {} --progress-bar off".format(" ".join(pip_instructions)))

    print("Install testing infra")
    assert 0 == run("pip install {} --progress-bar off".format(" ".join(other_dependencies)))

    # install einops
    assert 0 == run("pip install -e .")

    # we need to inform testing script which frameworks to use
    # this is done by setting a flag EINOPS_TEST_BACKENDS
    from tests import unparse_backends

    envvar_name, envvar_value = unparse_backends(backend_names=frameworks)
    return_code = run(
        "python -m pytest tests",
        **{envvar_name: envvar_value},
    )
    assert return_code == 0


if __name__ == "__main__":
    main()

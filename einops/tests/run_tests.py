"""
Runs tests that are appropriate for framework.
"""

import os
import sys
from subprocess import Popen
from pathlib import Path

__author__ = "Alex Rogozhnikov"


def run(cmd, **env):
    # keeps printing output when testing
    cmd = cmd.split(" ") if isinstance(cmd, str) else cmd
    print("running:", cmd)
    p = Popen(cmd, cwd=str(Path(__file__).parent), env={**os.environ, **env})
    p.communicate()
    return p.returncode


def main():
    _executable, *args = sys.argv
    frameworks = [x for x in args if x != "--pip-install"]
    pip_install_is_set = "--pip-install" in args
    framework_name2installation = {
        "numpy": ["numpy"],
        "torch": ["torch --index-url https://download.pytorch.org/whl/cpu"],
        "jax": ["jax[cpu]", "flax"],
        "tensorflow": ["tensorflow"],
        "cupy": ["cupy"],
        # switch to stable paddlepaddle, because of https://github.com/PaddlePaddle/Paddle/issues/63927
        # "paddle": ["paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html"],
        "paddle": ["paddlepaddle"],
        "oneflow": ["oneflow==0.9.0"],
        "pytensor": ["pytensor"],
    }

    usage = f"""
    Usage:   python -m einops.tests.run_tests <frameworks> [--pip-install]
    Example: python -m einops.tests.run_tests numpy pytorch --pip-install

    Available frameworks: {list(framework_name2installation)}
    When --pip-install is set, auto-installs requirements with pip.
     (make sure which pip points to right pip)
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

    if pip_install_is_set:
        print("Install testing infra")
        other_dependencies = ["pytest"]
        assert 0 == run("pip install {} --progress-bar off -q".format(" ".join(other_dependencies)))

        for framework in frameworks:
            print(f"Installing {framework}")
            pip_instructions = framework_name2installation[framework]
            assert 0 == run("pip install {} --progress-bar off -q".format(" ".join(pip_instructions)))

    # we need to inform testing script which frameworks to use
    # this is done by setting an envvar EINOPS_TEST_BACKENDS
    from einops.tests import unparse_backends

    envvar_name, envvar_value = unparse_backends(backend_names=frameworks)
    return_code = run(
        "python -m pytest .",
        **{envvar_name: envvar_value},
    )
    assert return_code == 0


if __name__ == "__main__":
    main()

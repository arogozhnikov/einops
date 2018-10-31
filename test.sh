#!/usr/bin/env bash

pip install -e .

# cupy is not skipped by default
# we need to run all tests twice - once for tensorflow eager, and once for
# those can't be run together
EINOPS_SKIP_CUPY=$EINOPS_SKIP_CUPY TF_EAGER=1 nosetests tests -vds
EINOPS_SKIP_CUPY=$EINOPS_SKIP_CUPY TF_EAGER=0 nosetests tests -vds

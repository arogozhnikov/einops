#!/usr/bin/env bash

pip install -e .
# we need to run all tests twice - once for tensorflow eager, and once for
# those can't be run together
TF_EAGER=1 nosetests tests
TF_EAGER=0 nosetests tests
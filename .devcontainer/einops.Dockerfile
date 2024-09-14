FROM python:3.11-bookworm AS einops-devimage
# note this will pull aarch64 or x86 depending on machine it is running on.
RUN pip install uv \
 && uv pip install --system pre-commit hatch
# auto pushes to github
pip install -r scripts/requirements-dev.txt \
&& mkdocs build \
&& mkdocs gh-deploy
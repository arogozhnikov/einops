# should be sourced when in einops folder

rm -f build/*
rm -f dist/*
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine upload dist/*
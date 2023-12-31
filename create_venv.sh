#!/bin/bash

venv_dir="$1"
python3 -m venv "$venv_dir"
echo "Virtual environment created at $venv_dir"


activate_script=""
if [[ $OSTYPE == "msys" || $OSTYPE == "cygwin" || $OSTYPE == "win32" ]]; then
    activate_script="$venv_dir/Scripts/activate"
else
    activate_script="$venv_dir/bin/activate"
fi
source "$1/bin/activate"

pip install -e .

pip -V

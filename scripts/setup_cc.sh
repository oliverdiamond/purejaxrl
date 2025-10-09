#!/bin/bash

# TODO: likely, we no longer need this venv in ~, since the idea of the home
# directory is to be lightweight, just for scheduling experiments etc. But, our
# experiment models are quite heavy weight, nullifying any potential benefits

export PY_VERSION="3.11"

module load "python/$PY_VERSION"

# make sure home folder has a venv
if [ ! -d ~/.venv ]; then
  echo "making a new virtual env in ~/.venv"
  python -m venv ~/.venv
  source ~/.venv/bin/activate
  echo "installing PyExpUtils"
  pip install PyExpUtils-andnp
fi

# Check if the `cpu` argument is provided and pass it to local_node_venv.sh
if [ "$1" == "cpu" ]; then
  echo "scheduling a job to install project dependencies (CPU mode)"
  sbatch --ntasks=1 --mem-per-cpu="12G" --export=ALL,path="$(pwd)" scripts/local_node_venv.sh cpu
else
  echo "scheduling a job to install project dependencies (GPU mode)"
  sbatch --ntasks=1 --mem-per-cpu="12G" --export=ALL,path="$(pwd)" scripts/local_node_venv.sh
fi

#!/bin/bash

#SBATCH --time=00:55:00
#SBATCH --account=def-amw8

module load "python/$PY_VERSION" rust
module load mujoco

# Check for argument and set requirements file accordingly
if [ "$1" == "cpu" ]; then
    requirements_file="requirements_cc_cpu.txt"
else
    requirements_file="requirements_cc.txt"
fi

cp $path/$requirements_file $SLURM_TMPDIR/
cd $SLURM_TMPDIR
python -m venv .venv
source .venv/bin/activate
pip install -r $requirements_file
pip install brax --no-index # This is nessesary when using the CC mujoco module
pip install envpool==0.8.4


# TODO: for some reason, pip cannot install any of the current wheels for this package.
# this is a pretty bad hack, but...
curl https://files.pythonhosted.org/packages/38/9c/3a3a831bfbd30fdedd61994d35df41fd0d47145693fe706976589214f811/connectorx-0.3.2-cp311-cp311-manylinux_2_28_x86_64.whl --output connectorx-0.3.2-cp311-cp311-linux_x86_64.whl
pip install connectorx-0.3.2-cp311-cp311-linux_x86_64.whl

# Set the tar file name based on the argument
if [ "$1" == "cpu" ]; then
    tar_file="venv_cpu.tar.xz"
else
    tar_file="venv.tar.xz"
fi

tar -cavf $tar_file .venv
cp $tar_file $path/

pip freeze

#!/bin/bash

set +x
set -eo pipefail

ENV_NAME="${ENV_NAME:-amalia-eval}"

# check if CONDA_HOME is set
if [ -z "$CONDA_HOME" ]
then
    echo "Please set CONDA_HOME to the location of your conda installation"
    exit 1
fi

# Conda envs may not be in the conda home directory
CONDA_ENVS="${CONDA_ENVS:-$CONDA_HOME/envs}"

echo "Env name: $ENV_NAME"
echo "Conda home: $CONDA_HOME"
echo "Conda envs: $CONDA_ENVS"

# get the directory of this script, which should be the root directory of the project
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source ${CONDA_HOME}/etc/profile.d/conda.sh

# create the conda environment
conda create -n "$ENV_NAME" python=3.12 -y

# check if the environment was created in the expected directory
if [ ! -d "${CONDA_ENVS}/${ENV_NAME}" ]
then
    echo "Environment $ENV_NAME not found at ${CONDA_ENVS}/${ENV_NAME}."
    echo "If you are storing your conda environments in a different location, please set the CONDA_ENVS environment variable to the location of your conda environments."
    exit 1
fi

# activate the conda environment
conda activate "$ENV_NAME"

conda install -y -c "nvidia/label/cuda-12.4.0" cuda-toolkit cuda-nvcc cudnn

export PATH=${CONDA_ENVS}/${ENV_NAME}/bin:$PATH
export LD_LIBRARY_PATH=${CONDA_ENVS}/${ENV_NAME}/lib:$LD_LIBRARY_PATH
export CUDA_HOME=${CONDA_ENVS}/${ENV_NAME}

echo "Installing dependencies..."
pip install -e .

echo ""
echo "Environment creation completed successfully!"
echo ""
echo "Do you want to pack this environment for HPC deployment?"
echo "This will install conda-pack and create a ${OUTPUT_FILE} file."
read -p "Pack environment? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Install conda-pack if not already installed
    echo "Installing conda-pack..."
    conda install conda-pack -c conda-forge -y

    OUTPUT_FILE="${ENV_NAME}.tar.gz"

    # Pack the environment
    echo "Packing environment (this may take a few minutes)..."
    conda pack -n ${ENV_NAME} -o ${OUTPUT_FILE}

    # Get file size
    FILE_SIZE=$(du -h ${OUTPUT_FILE} | cut -f1)

    echo ""
    echo "================================================"
    echo "SUCCESS! Environment packed successfully"
    echo "================================================"
    echo "Output file: ${OUTPUT_FILE}"
    echo "File size: ${FILE_SIZE}"
    echo ""
    echo "Next steps:"
    echo "1. Transfer the file to your HPC cluster:"
    echo "   scp ${OUTPUT_FILE} username@hpc-cluster:/path/to/destination/"
    echo ""
    echo "2. Run remote_setup.sh on the HPC cluster"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "SUCCESS! Environment created successfully"
    echo "================================================"
    echo "Environment name: ${ENV_NAME}"
    echo ""
    echo "The environment is ready to use locally."
    echo "If you want to pack it later for HPC deployment, you can run:"
    echo "conda activate ${ENV_NAME}"
    echo "conda install conda-pack -c conda-forge"
    echo "conda pack -n ${ENV_NAME} -o ${OUTPUT_FILE}"
    echo "================================================"
fi
#!/bin/bash


conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda102  # or [ magma-cuda101 | magma-cuda100 | magma-cuda92 ] depending on your cuda version


git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(10.2))/../"}
python setup.py install

cd /home/michaeldeleo/Workspace/HaliteIV-AI/pytorch/third_party/nccl/nccl && env CCACHE_DISABLE=1 SCCACHE_DISABLE=1 make CXX=/usr/bin/c++ CUDA_HOME=/usr/lib/cuda NVCC=/usr/lib/cuda/bin/nvcc NVCC_GENCODE=-gencode=arch=compute_75,code=sm_75 BUILDDIR=/home/michaeldeleo/Workspace/HaliteIV-AI/pytorch/build/nccl VERBOSE=0 -j && /home/michaeldeleo/anaconda3/envs/HaliteIV-AI/bin/cmake -E touch /home/michaeldeleo/Workspac
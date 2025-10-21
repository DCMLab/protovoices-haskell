#!/bin/sh

# VERSION="2.3.1"
# ROCM_VERSION="6.0"
VERSION="2.9.0"
ROCM_VERSION="6.4"
CUDA_VERSION="130"

case $1 in
    cpu)
        filename="libtorch-shared-with-deps-${VERSION}+cpu.zip"
        url="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${VERSION}%2Bcpu.zip"
        ;;
    cuda)
        filename="libtorch-shared-with-deps-${VERSION}+cuda${CUDA_VERSION}.zip"
        url="https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-shared-with-deps-${VERSION}%2Bcu${CUDA_VERSION}.zip"
        ;;
    rocm)
        filename="libtorch-shared-with-deps-${VERSION}+rocm${ROCM_VERSION}.zip"
        url="https://download.pytorch.org/libtorch/rocm${ROCM_VERSION}/libtorch-shared-with-deps-${VERSION}%2Brocm${ROCM_VERSION}.zip"
        ;;
    *)
        echo "Pick cpu, cuda, or rocm!"
        filename=""
        url=""
        ;;
esac

if [ "$filename" != "" ]; then
    if [ ! -f "$filename" ]; then
        curl "$url" > "$filename"
    fi
    if [ -d "./libtorch" ]; then
        rm -r libtorch
    fi
    unzip "$filename"
fi

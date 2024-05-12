#!/bin/sh

VERSION="2.0.0"
ROCM_VERSION="5.4.2"
CUDA_VERSION="118"

case $1 in
    cpu)
        filename="libtorch-cxx11-abi-shared-with-deps-${VERSION}+cpu.zip"
        url="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Bcpu.zip"
        ;;
    cuda)
        filename="libtorch-cxx11-abi-shared-with-deps-${VERSION}+cuda.zip"
        url="https://download.pytorch.org/libtorch/cu${CUDA_VERSION}/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Bcu${CUDA_VERSION}.zip"
        ;;
    rocm)
        filename="libtorch-cxx11-abi-shared-with-deps-${VERSION}+rocm.zip"
        url="https://download.pytorch.org/libtorch/rocm${ROCM_VERSION}/libtorch-cxx11-abi-shared-with-deps-${VERSION}%2Brocm${ROCM_VERSION}.zip"
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

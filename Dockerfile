
# Stage 1: Start with NVIDIA CUDA image as base
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as cuda-base

# Stage 2: Micromamba Image
FROM mambaorg/micromamba:1.4.9 as micromamba-base
USER root

# Set Python-related environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt-get update
RUN apt-get install -y --reinstall build-essential

# Copy over CUDA libraries from the CUDA base image
COPY --from=cuda-base /usr/local/cuda /usr/local/cuda

# Ensure CUDA libraries are in the correct path
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

WORKDIR /app
COPY . /app

RUN micromamba install -y -n base -f /app/env.yml && micromamba clean --all --yes

RUN git config --global --add safe.directory /app


FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/program

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml && \
    conda clean -afy

# Copy the model code
COPY . .

# Set environment variables
ENV PYTHONPATH=/opt/program
ENV EXPERIMENT_PATH=/opt/ml/model
ENV OPENBLAS_NUM_THREADS=1

# Activate conda environment and set the entrypoint
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate pix2poly && python api.py"] 
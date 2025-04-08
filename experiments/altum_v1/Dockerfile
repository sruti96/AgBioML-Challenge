FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    wget

# Get conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
# Remove the downloaded installer
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to the PATH environment variable for all subsequent RUN instructions
ENV PATH=/opt/conda/bin:$PATH

# Now conda commands should work directly
# First, define an environment with scikit-learn
RUN conda create -y -n sklearn-env python=3.12 scikit-learn pandas seaborn

# Then define an environment with torch
RUN conda create -y -n torch-env python=3.12 pytorch::pytorch pytorch::torchvision pandas seaborn

# Then define an environment with tensorflow
RUN conda create -y -n tensorflow-env  -c conda-forge pandas seaborn
RUN conda run -n tensorflow-env pip install 'tensorflow[and-cuda]' keras

# Then define an environment with scanpy
RUN conda create -y -n scanpy-env python=3.12 pandas seaborn
# Use conda run to execute pip install within the scanpy-env
RUN conda run -n scanpy-env pip install 'scanpy[leiden]'
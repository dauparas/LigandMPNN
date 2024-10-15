FROM nvidia/cuda:12.1.0-base-ubuntu22.04
COPY . /home/LigandMPNN

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && apt-get install python3 -y \
    && apt-get install python3-pip -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install CUDA compatibility libraries
RUN apt-get update \
    && apt-get install -y cuda-compat-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set the library path
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
ENV PYTHONPATH=/home/LigandMPNN/
ENV LIGAND_MPNN=/home/LigandMPNN/
RUN pip install -r /home/LigandMPNN/requirements.txt

WORKDIR /home/LigandMPNN/
RUN bash get_model_params.sh "./model_params"

# ------------------------------------------------------------------ #
#  DQN CartPole — Dockerfile
#
#  Uses a standard Python base image and installs PyTorch via pip.
#  Automatically trains on GPU if available, falls back to CPU.
#
#  Build:
#    docker build -t dqn-cartpole .
#
#  Run with GPU:
#    docker run --gpus all -v $(pwd)/results:/app/results dqn-cartpole
#
#  Run CPU only:
#    docker run -v $(pwd)/results:/app/results dqn-cartpole
# ------------------------------------------------------------------ #

# Standard Python 3.10 slim image — lightweight and universally available
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# --- System dependencies ---
# libgl1 and libglib2.0 are required by pygame/gymnasium for rendering
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch installation ---
# Install PyTorch CPU version via the official pip index.
# At runtime, agent.py automatically switches to CUDA if available:
#   self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# To use GPU, rebuild with the CUDA index instead:
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 \
#                -t dqn-cartpole .
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX}

# --- Python dependencies ---
# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir \
    "gymnasium[classic-control]>=0.29.0" \
    matplotlib>=3.7.0 \
    numpy>=1.24.0 \
    Pillow>=10.0.0 \
    pyyaml>=6.0.0 \
    idna>=2.8 \
    requests>=2.31

# --- Project files ---
COPY . .

# Create output directories in case they don't exist
RUN mkdir -p results assets

# --- Default command ---
# Running the container starts training by default.
# Override with: docker run <image> python src/record_demo.py
CMD ["python", "src/train.py"]
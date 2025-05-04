FROM python:3.9-slim

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install OS deps for matplotlib & torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgl1-mesa-glx \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create & use workspace directory
WORKDIR /workspace

# Copy in requirements and install Python packages
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Default to bash
CMD ["bash"]

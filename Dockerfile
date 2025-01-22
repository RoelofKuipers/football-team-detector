# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files/folders
COPY main.py .
COPY src/ src/
COPY checkpoints/*.pt checkpoints/

# Create data directory for mounting
RUN mkdir /data /output

ENTRYPOINT ["python", "main.py"]

# Set performance environment variables
ENV OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 \ 
    YOLO_VERBOSE=False


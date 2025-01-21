# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files/folders
COPY main.py .
COPY src/ src/
COPY checkpoints/*.pt checkpoints/

# Create necessary directories
RUN mkdir -p data output checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
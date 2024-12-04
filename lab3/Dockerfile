FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy service files
COPY lab3/service.py lab3/bentofile.yaml lab3/raw_xgb.onnx  ./

# Install Python dependencies
RUN pip install bentoml pandas numpy onnx onnxruntime python-dateutil

# Expose the port
EXPOSE 3000

# Start BentoML service
CMD ["bentoml", "serve", "service:svc", "--production"]

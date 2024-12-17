FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Install necessary dependencies
RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.4.0 mamba-ssm==2.2.2 timm==1.0.9 einops transformers opencv-python scipy flask python-dotenv pymongo matplotlib scikit-image scikit-learn wandb elasticsearch seaborn albumentations fightingcv-attention positional-encodings[pytorch,tensorflow]

# Install additional dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set working directory and copy application code
WORKDIR /app
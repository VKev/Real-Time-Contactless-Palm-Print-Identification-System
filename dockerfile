FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

# Install necessary dependencies
RUN pip install --no-cache-dir tensorboardX causal-conv1d==1.4.0 mamba-ssm==2.2.2 "timm>=0.9.5,<0.10.0" einops transformers opencv-python scipy flask python-dotenv pymongo matplotlib scikit-image scikit-learn wandb elasticsearch seaborn albumentations fightingcv-attention positional-encodings[pytorch,tensorflow] pytorch_lightning tensorflow

# Install additional dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Set working directory and copy application code
WORKDIR /app

#docker run -it --gpus all --name mamba-environment -p 8081:8081 -v C:\Vkev\Repos\Mamba-Environment:/app -w /app mamba-environment

#docker run -it --gpus all --name mamba-environment -p 8081:8081 -v /var/run/docker.sock:/var/run/docker.sock -v /c/Vkev/Repos/Mamba-Environment:/app -e DOCKER_HOST=unix:///var/run/docker.sock -w /app mamba-environment

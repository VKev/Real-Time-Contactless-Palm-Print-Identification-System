docker run -it --shm-size=8g --gpus all --name khanghv -p 8081:8081 -v C:\Vkev\Repos\Mamba-Environment:/app -w /app vkev25811/cuda12.4-cudnn9-devel:latest

docker pull nvcr.io/nvidia/tritonserver:25.03-py3

docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v C:\Vkev\Repos\Mamba-Environment\Palm-Print-Identification-System-V2\model_repository:/models nvcr.io/nvidia/tritonserver:25.03-py3 tritonserver --model-repository=/models

pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git
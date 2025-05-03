# Palm-Print-Identification-System

## Prerequisites

Before starting, ensure you have **Docker** installed:

Additionally, you will need to download a model checkpoint from Google Drive. Ensure you can access the provided link.

## Demo Instructions

### 1. Setup

First, clone the repository:
```bash
git clone https://github.com/VKev/Palm-Print-Identification-System-V2.git
```

Next, download the model checkpoint from: [Here](https://drive.google.com/file/d/1h28z9Es4IRkCnJTiPyqy41-qHPHrLH8Z/view?usp=sharing)

After downloading, place the model.pt file in the following directory structure:
```
model_repository/
└── 1/
    └── model.pt
```
**Note**: If the model_repository/1/ directories do not exist, create them manually.

Then, install the required Python packages:
```bash
pip install -r requirements.txt
```
**Tip**: It is recommended to use a virtual environment to avoid conflicts with system packages.

#### 2. Run

Ensure Docker is running on your system. Then, start the Docker services:
```bash
docker-compose up -d
```

After the services are up, launch the application:
```bash
python app.py
```
Once the application is running, access the demo at:
`http://localhost:7000`


## Train and Test

To train and test the model, you need to obtain the dataset first.

Request the dataset from the following links:

- [IITD Palmprint Database](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm)

- [PolyU 3D Palmprint Database](https://www4.comp.polyu.edu.hk/~csajaykr/myhome/database_request/3dhand/Hand3D.htm)

After requesting, send an email to huynhkhang7452@gmail.com with proof of your request acceptance. The dataset will be sent to you within 1-2 days. Note that the dataset includes additional public datasets and is formatted for use with this code.

Once you receive the dataset, extract it to a directory of your choice, for example, /path/to/dataset.

You can train and test the model using either the local Python environment or Docker.

#### Using Local Python Environment

Navigate to the feature_extraction directory, and install the required packages:
```bash
cd path/to/Palm-Print-Identification-System-V2/feature_extraction

pip install -r requirements.txt
```
Run the training script:
```
python train.py --train_path /path/to/dataset/TrainAndTest/train --test_path /path/to/dataset/TrainAndTest/test
```
Replace /path/to/dataset with the actual path where you extracted the dataset.

The checkpoints will be saved in the checkpoints/ directory within feature_extraction/.

To resume training from a checkpoint:

```bash
python train.py --train_path /path/to/dataset/TrainAndTest/train --test_path /path/to/dataset/TrainAndTest/test --checkpoint_path checkpoints/your_checkpoint_want_to_resume.pth
```

To test the model:

```bash
python test.py --validate_path /path/to/dataset/TrainAndTest/test --checkpoint_path checkpoints/your_checkpoint_want_to_test.pth
```

#### Using Docker

Pull the Docker image:
```bash
docker pull vkev25811/cuda12.4-cudnn9-devel:latest
```

Run the Docker container, mounting the repository and the dataset directories:

```bash
docker run -it --shm-size=8g --gpus all --name palm_print_container -p 8081:8081 -v /path/to/Palm-Print-Identification-System-V2:/app -v /path/to/dataset:/dataset -w /app vkev25811/cuda12.4-cudnn9-devel:latest
```
Replace /path/to/Palm-Print-Identification-System-V2 with the actual path to the cloned repository, and /path/to/dataset with the path where you extracted the dataset.

Run the training script:
```
python train.py --train_path /path/to/dataset/TrainAndTest/train --test_path /path/to/dataset/TrainAndTest/test
```
Replace /path/to/dataset with the actual path where you extracted the dataset.

The checkpoints will be saved in the checkpoints/ directory within feature_extraction/.

To resume training from a checkpoint:

```bash
python train.py --train_path /path/to/dataset/TrainAndTest/train --test_path /path/to/dataset/TrainAndTest/test --checkpoint_path checkpoints/your_checkpoint_want_to_resume.pth
```

To test the model:

```bash
python test.py --validate_path /path/to/dataset/TrainAndTest/test --checkpoint_path checkpoints/your_checkpoint_want_to_test.pth
```

## Convert checkpoint .pth → .pt for Demo

Navigate to the TorchScript helper:

```bash
cd Palm-Print-Identification-System-V2/feature_extraction
```

Open torchscript.py and update checkpoint_path to your desired .pth, then run the converter:

```bash
python torchscript.py
```
This will create model_repository/1/model.pt automatically.

Start (or restart) the demo services:
```bash
docker-compose up -d
```
If Docker services were already running, clean them up first before start services:
```bash
docker-compose down
```


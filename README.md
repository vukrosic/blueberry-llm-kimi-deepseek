# Blueberry LLM 🫐

A Mixture of Experts (MoE) language model implementation with Docker support for easy cloud GPU deployment.

## Quick Start with Docker

### Prerequisites
- Docker with NVIDIA GPU support
- NVIDIA Docker runtime

### One-Command Training
```bash
# Clone and start training immediately
git clone <your-repo>
cd blueberry
chmod +x train.sh
./train.sh train
```

### Available Commands
```bash
./train.sh train           # Start training
./train.sh dev             # Development environment  
./train.sh tensorboard     # Launch TensorBoard
./train.sh build           # Build Docker image
./train.sh clean           # Clean up resources
```

## Cloud GPU Deployment

### AWS EC2 with GPU
```bash
# Launch GPU instance (g4dn.xlarge or p3.2xlarge)
# Install Docker + NVIDIA Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone and train
git clone <your-repo>
cd blueberry
./train.sh train
```

### Google Cloud Platform
```bash
# Create GPU VM with Deep Learning image
gcloud compute instances create blueberry-train \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --maintenance-policy=TERMINATE

# SSH and run
gcloud compute ssh blueberry-train
git clone <your-repo>
cd blueberry
./train.sh train
```

### Azure
```bash
# Create GPU VM
az vm create \
  --resource-group myResourceGroup \
  --name blueberry-vm \
  --image Canonical:UbuntuServer:18.04-LTS:latest \
  --size Standard_NC6 \
  --admin-username azureuser \
  --generate-ssh-keys

# SSH and setup (similar to AWS steps)
```

## Model Architecture

- **Mixture of Experts**: 8 experts, top-2 routing
- **Architecture**: 384d model, 6 layers, 8 heads
- **Training**: Muon optimizer with gradient accumulation
- **Data**: SmolLM corpus (cosmopedia-v2)

## Configuration

Edit the `MoEModelConfig` in `llm.py` to customize:
- Model size (`d_model`, `n_layers`, `n_heads`)
- MoE parameters (`num_experts`, `expert_top_k`)
- Training settings (`batch_size`, `max_steps`, `muon_lr`)

## Monitoring

- **TensorBoard**: `./train.sh tensorboard` → http://localhost:6006
- **Weights & Biases**: `./train.sh train --wandb`

## Manual Docker Commands

```bash
# Build
docker build -t blueberry-llm .

# Run with GPU
docker run --gpus all -v $(pwd)/data_cache:/app/data_cache blueberry-llm

# Development
docker run --gpus all -it -v $(pwd):/app blueberry-llm /bin/bash
```

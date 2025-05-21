# LitBenchReward Model Training

This repository contains scripts to train different types of reward models for preference learning. The training automatically adjusts batch sizes based on available GPUs to maintain a consistent effective batch size.

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPUs (for faster training)
- Hugging Face account with access to the model you want to use

### Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Required for accessing gated models
   export HF_TOKEN="your_huggingface_token"
   
   # Optional: Configure WANDB for logging
   export WANDB_API_KEY="your_wandb_api_key"
   export WANDB_PROJECT="reward_model_training"
   ```

## Model Types

This repository supports training two types of reward models:

- **Bradley-Terry Reward Model (BTRM)**: A pairwise preference model that learns from comparisons between chosen and rejected content. It uses a classification approach that estimates the probability of one output being preferred over another.

- **Generative Reward Model (GenRM)**: A generative approach to preference learning where the model learns to generate reasoning about preferences along with the final verdict. It uses a completion-only fine-tuning approach on rationales.

## Available Training Scripts

This repository includes several training scripts:

- **train_BTRM.sh**: Trains a Bradley-Terry Reward Model for pairwise preference learning
- **train_GenRM.sh**: Trains a Generative Reward Model using the LitBench-Rationales dataset

### Configuration

Before running a training script, modify the following:

1. In the bash scripts (`train_BTRM.sh` or `train_GenRM.sh`):
   - Change the base model to your preferred model (default: "meta-llama/Llama-3.2-1B" or "meta-llama/Llama-3.2-1B-Instruct")
   - Adjust `TARGET_BATCH_SIZE` if needed (default: 128)
   - Modify `OUTPUT_DIR` path if needed
   - Adjust training parameters like `--epochs`, `--max_length`, and `--learning_rate`

2. In the Python scripts (if needed):
   - Dataset paths or sources
   - Advanced model configuration
   - Evaluation settings

### Running the Training

1. Make the script executable:
   ```bash
   chmod +x train_BTRM.sh  # or chmod +x train_GenRM.sh
   ```

2. Run the training script:
   ```bash
   ./train_BTRM.sh  # or ./train_GenRM.sh
   ```

The script will:
1. Detect available GPUs
2. Calculate optimal batch size and gradient accumulation steps
3. Train the model with the specified effective batch size
4. Save the model to the specified output directory

## Key Features

- **Adaptive batch sizing**: Automatically adjusts batch size and gradient accumulation steps
- **Timestamp-based output directories**: Each run creates a uniquely named output folder
- **Distributed training support**: Leverages multiple GPUs efficiently
- **Weights & Biases integration**: Tracks metrics during training
- **Checkpointing**: Saves model checkpoints during training

## Troubleshooting

- **Out of memory errors**: Reduce `TARGET_BATCH_SIZE` in the bash script
- **Model access issues**: Ensure your `HF_TOKEN` is set and has access to the model
- **CUDA errors**: Make sure you have compatible NVIDIA drivers installed

## License

[License information here]

## Acknowledgments

This project uses datasets from Hugging Face Hub and builds on the TRL library for reward modeling.

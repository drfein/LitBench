# LitBench Reward Model Training

This repository contains scripts to train different types of reward models for preference learning. The training automatically adjusts batch sizes based on available GPUs to maintain a consistent effective batch size.

### Quick Start with Rehydration

1. **Rehydrate the dataset** (takes 1-2 hours, but significantly improves data quality):
   ```bash
   python scripts/rehydrate.py
   ```

2. **Run training as usual** - scripts will automatically detect and use the rehydrated data:
   ```bash
   ./training/train_BTRM.sh
   ```

### Why Rehydrate?

- **96%+ completeness**: Enhanced dataset with 425 additional recovered comment IDs
- **Fresh data**: Latest content directly from Reddit API
- **Better training**: More complete dataset leads to improved model performance
- **Automatic integration**: Training scripts seamlessly use rehydrated data when available

### Rehydration Details

The rehydration process:
1. Loads the enhanced comment ID dataset with recovered comment IDs
2. Fetches ~3,400 unique comment stories from Reddit API
3. Creates `data/rehydrated_test_data.csv` with complete story text and metadata
4. Training scripts use this rehydrated data (required for testing)

**Requirements for rehydration:**
- Reddit API access (free - no special permissions needed)
- Internet connection
- 1-2 hours of patience (due to Reddit rate limits)

Reddit API credentials can be configured via:
- Environment variables: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`
- Or the credentials are embedded in the script for convenience

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-compatible GPUs (for faster training)
- Hugging Face account with access to the model you want to use
- Internet connection for dataset rehydration (required)

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

## Recommended Workflow

### Required: Rehydrate the Dataset

**Rehydration is now required** for test data:
```bash
python scripts/rehydrate.py
```
This creates `data/rehydrated_test_data.csv` with fresh, complete data from Reddit.

### Train Your Model

Once rehydrated data exists, train your model:
```bash
./training/train_BTRM.sh  # or ./training/train_GenRM.sh
```

The training scripts will automatically use the rehydrated test data.

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

1. In the bash scripts (`training/train_BTRM.sh` or `training/train_GenRM.sh`):
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
   chmod +x training/train_BTRM.sh  # or chmod +x training/train_GenRM.sh
   ```

2. Run the training script:
   ```bash
   ./training/train_BTRM.sh  # or ./training/train_GenRM.sh
   ```

The script will:
1. Detect available GPUs
2. Calculate optimal batch size and gradient accumulation steps
3. **Automatically use rehydrated data if available**
4. Train the model with the specified effective batch size
5. Save the model to the specified output directory

## Rehydration Advanced Usage

### Custom Rehydration Options

```bash
# Custom output directory
python scripts/rehydrate.py --output_dir ./custom_data

# Adjust worker threads (default: 3, for Reddit rate limits)
python scripts/rehydrate.py --max_workers 2

# Check rehydration status
ls -la data/rehydrated_test_data.csv
```

### Data Quality Information

| Dataset Version | Completeness | Comment IDs | Description |
|----------------|--------------|-------------|-------------|
| **Rehydrated** | **96%+** | **2,381 complete** | **Fresh from Reddit, enhanced with recovered IDs** |

### Rehydration Process Details

The rehydration uses an enhanced comment ID dataset that includes:
- 425 additional comment IDs recovered through intelligent text matching
- 100% verified accuracy (all recovered IDs validated)
- Only complete rows (both chosen and rejected stories present)

## Key Features

- **🚀 Dataset Rehydration**: Fetch fresh, complete data directly from Reddit
- **🔄 Automatic Data Detection**: Training scripts seamlessly use rehydrated data
- **Adaptive batch sizing**: Automatically adjusts batch size and gradient accumulation steps
- **Timestamp-based output directories**: Each run creates a uniquely named output folder
- **Distributed training support**: Leverages multiple GPUs efficiently
- **Weights & Biases integration**: Tracks metrics during training
- **Checkpointing**: Saves model checkpoints during training

## File Structure

```
LitBench/
├── src/                     # Core library code
│   ├── __init__.py         # Package initialization
│   ├── dataloader.py       # Smart dataloader (auto-detects rehydrated data)
│   └── reddit_utils.py     # Reddit API utilities
├── scripts/                # Standalone scripts
│   ├── rehydrate.py        # Dataset rehydration script
│   ├── train_BT.py         # Bradley-Terry training script
│   └── train_GenRM.py      # Generative reward model training
├── training/               # Training configuration scripts
│   ├── train_BTRM.sh      # Bradley-Terry training runner
│   └── train_GenRM.sh     # GenRM training runner
├── data/                   # Data directory
│   ├── rehydrated_test_data.csv  # Created by rehydration (ignored by git)
│   └── processed/              # Tokenized datasets cache
├── README.md               # Documentation
├── requirements.txt        # Dependencies (includes Reddit API support)
└── .gitignore             # Git ignore rules
```

## Troubleshooting

### General Issues
- **Out of memory errors**: Reduce `TARGET_BATCH_SIZE` in the bash script
- **Model access issues**: Ensure your `HF_TOKEN` is set and has access to the model
- **CUDA errors**: Make sure you have compatible NVIDIA drivers installed

### Rehydration Issues
- **Reddit API errors**: The script includes automatic retry logic and rate limiting
- **Slow rehydration**: Normal! Reddit rate limits require 1-2 hours for full dataset
- **Rehydration fails**: Training will automatically fall back to original HuggingFace datasets
- **Network issues**: Rehydration can be resumed - it saves progress periodically

### Data Issues
- **Missing rehydrated data**: Run `python scripts/rehydrate.py` to create it
- **Want to skip rehydration**: Just run training directly - it works with original data too
- **Data directory missing**: Created automatically on first training run

## Performance Notes

- **Rehydrated datasets typically improve model performance** due to more complete training data
- **Rehydration is optional** - everything works with original datasets too
- **First rehydration takes time** but subsequent training runs are fast
- **Rehydrated data is cached** and reused across multiple training runs


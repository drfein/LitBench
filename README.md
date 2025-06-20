# LitBench Reward Model Training

## Quick Setup Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd LitBench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials

Set up your HuggingFace token:
```bash
export HF_TOKEN="your_huggingface_token"
```

Set up Reddit API credentials (required for dataset creation):
```bash
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
```
*A guide to creating credentials can be found [here](https://docs.google.com/document/d/19o3O_lMsi3i8TNgCayYGUs6TRZu6Gj7RgYmJ9gSLiBY/edit?usp=sharing)*

Optional: Configure Weights & Biases:
```bash
export WANDB_API_KEY="your_wandb_api_key"
export WANDB_PROJECT="reward_model_training"
```

### 3. Rehydrate Test Set

Rehydrate the test dataset from Reddit:
```bash
python scripts/rehydrate.py
```
*This takes 1-2 hours due to Reddit rate limits.*

### 4. Train Models

Make training scripts executable:
```bash
chmod +x training/train_BTRM.sh
chmod +x training/train_GenRM.sh
```

Train Bradley-Terry Reward Model:
```bash
./training/train_BTRM.sh
```

Or train Generative Reward Model:
```bash
./training/train_GenRM.sh
```

### 5. Configuration (Optional)

Edit `training/train_BTRM.sh` or `training/train_GenRM.sh` to modify:
- Base model (default: `meta-llama/Llama-3.2-1B`)
- Batch size (default: 128 effective batch size)
- Output directory
- Training parameters

That's it! Your trained model will be saved to the specified output directory.


import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, load_dataset
import wandb
import argparse
import os
from tqdm import tqdm
import numpy as np
import datetime
import random
import traceback
import sys
from trl import RewardTrainer, RewardConfig
from trl.trainer.utils import RewardDataCollatorWithPadding
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from dataloader import LitBenchDataLoader

try:
    from transformers.file_utils import WEIGHTS_NAME
except ImportError:
    try:
        from transformers.utils import WEIGHTS_NAME
    except ImportError:
        WEIGHTS_NAME = "pytorch_model.bin"

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a Bradley-Terry reward model')
parser.add_argument('--base_model', type=str, default="meta-llama/Llama-3.2-1B", help='Base model name from Hugging Face')
parser.add_argument('--output_dir', type=str, default='./bt_reward_model', help='Directory to save model and predictions')
parser.add_argument('--data_dir', type=str, default='./data', help='Directory containing the datasets')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=2, help='Total batch size')
parser.add_argument('--max_length', type=int, default=2048, help='Max sequence length')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help='Number of steps to accumulate gradients')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--keep_checkpoints', type=int, default=3, help='Number of best checkpoints to keep')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint for resuming training')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
args = parser.parse_args()

# Create all necessary directories early to avoid serialization errors
def ensure_dir_exists(directory):
    """Create directory if it doesn't exist and make sure it's writeable"""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
            test_file = os.path.join(directory, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            print(f"Error creating or writing to directory {directory}: {e}")
            if args.local_rank <= 0:
                print("WARNING: Directory access issue may cause training failure!")
    return os.path.exists(directory)

ensure_dir_exists(args.output_dir)
ensure_dir_exists(args.data_dir)
ensure_dir_exists(os.path.join(args.data_dir, 'processed'))
ensure_dir_exists(os.path.join(args.data_dir, 'hf_cache'))

num_gpus = torch.cuda.device_count()

# Initialize process group for distributed training
if args.local_rank != -1:
    torch.distributed.init_process_group(backend="nccl")
    is_main_process = args.local_rank == 0
else:
    is_main_process = True

# Initialize Weights and Biases - only in the main process
if is_main_process:
    wandb.login()
    wandb.init(project="bt_reward_model_training")
    print(f"Training with {num_gpus} GPUs, local_rank: {args.local_rank}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective total batch size: {args.batch_size * args.gradient_accumulation_steps}")

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = args.base_model
MAX_LENGTH = args.max_length

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if is_main_process:
    print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if is_main_process:
            print(f"Setting pad_token to eos_token: {tokenizer.pad_token}, id: {tokenizer.pad_token_id}")
except Exception as e:
    if is_main_process:
        print(f"Error loading tokenizer: {e}")
        traceback.print_exc()
    sys.exit(1)

if is_main_process:
    print(f"Available memory before dataset loading: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated, {torch.cuda.memory_reserved()/1024**2:.2f}MB reserved")

try:
    # Initialize and use our custom LitBench dataloader
    cache_dir = os.path.join(args.data_dir, 'hf_cache')
    if is_main_process:
        print("Initializing LitBench DataLoader...")
    
    litbench_loader = LitBenchDataLoader(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        cache_dir=cache_dir,
        data_dir=args.data_dir,
        use_fast_tokenizer=True
    )
    
    if is_main_process:
        print("Loading and processing datasets using LitBench DataLoader...")
    
    model_name_safe = MODEL_NAME.replace('/', '_')
    train_dataset, val_dataset, test_dataset = litbench_loader.prepare_datasets(model_name_safe)
    
    if is_main_process:
        print(f"Datasets prepared: {len(train_dataset)} examples in train, {len(val_dataset)} examples in val, {len(test_dataset)} examples in test")
        print(f"Available memory after dataset loading: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated, {torch.cuda.memory_reserved()/1024**2:.2f}MB reserved")
except Exception as e:
    if is_main_process:
        print(f"Error loading datasets: {e}")
        traceback.print_exc()
    sys.exit(1)

import gc
gc.collect()
torch.cuda.empty_cache()
if is_main_process:
    print(f"Memory after tokenization and GC: {torch.cuda.memory_allocated()/1024**2:.2f}MB allocated, {torch.cuda.memory_reserved()/1024**2:.2f}MB reserved")

run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"{model_name_safe}_{run_timestamp}"

compute_dtype = torch.bfloat16
if is_main_process:
    print(f"Using {compute_dtype} for computation")

# Configure TRL RewardConfig for training
reward_config_args = {
    "output_dir": args.output_dir,
    "per_device_train_batch_size": args.batch_size,
    "per_device_eval_batch_size": args.batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "gradient_checkpointing": False,
    "learning_rate": args.learning_rate,
    "report_to": ["wandb"] if is_main_process else [],
    "num_train_epochs": args.epochs,
    "fp16": False,
    "bf16": True,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 3,
    "warmup_ratio": 0.1,
    "remove_unused_columns": False,
    "run_name": run_id,
    "load_best_model_at_end": False,
    "max_grad_norm": 1.0,
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
    "local_rank": args.local_rank,
    "ddp_bucket_cap_mb": 25,
    "tf32": True,
    "no_cuda": False,
}

if args.local_rank != -1:
    reward_config_args.update({
        "ddp_find_unused_parameters": False,
        "ddp_broadcast_buffers": False,
        "torch_compile": False,
    })

reward_config = RewardConfig(**reward_config_args)

if is_main_process:
    print("Loading base model...")

is_distributed = args.local_rank != -1

# Load model with model parallelism
if is_main_process:
    print("Loading model with model parallelism...")

# Initialize model with model parallelism
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=compute_dtype,
    device_map="auto",  # Automatically distribute across available GPUs
    low_cpu_mem_usage=True,
    num_labels=1
)

for param in model.parameters():
    param.requires_grad = True

model.config.pad_token_id = tokenizer.pad_token_id
if is_main_process:
    print(f"Set model's pad_token_id to {model.config.pad_token_id}")

data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)


trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

if is_main_process:
    print("Starting training...")
try:
    os.makedirs(args.output_dir, exist_ok=True)
    test_file = os.path.join(args.output_dir, ".write_test")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        if is_main_process:
            print(f"Verified output directory is writeable: {args.output_dir}")
    except Exception as e:
        if is_main_process:
            print(f"WARNING: Output directory {args.output_dir} is not writeable: {e}")
            print("This may cause training to fail when saving checkpoints")
    if args.resume_from_checkpoint:
        if is_main_process:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        if not os.path.exists(args.resume_from_checkpoint):
            if is_main_process:
                print(f"WARNING: Checkpoint path does not exist: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        if is_main_process:
            print("Starting training from scratch (not resuming from checkpoint)")
        trainer.train()
    trainer.save_model(args.output_dir)
    if is_main_process:
        print(f"Final model saved to {args.output_dir}")
except Exception as e:
    if is_main_process:
        print(f"Training failed with error: {str(e)}")
        traceback.print_exc()
    if is_main_process:
        wandb.finish()
    sys.exit(1)

if is_main_process:
    wandb.finish()

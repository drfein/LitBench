import argparse
import os
from typing import Dict, List, Optional
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from dataloader import SFTDataLoaderCOT, SFTDataLoaderDirect
import wandb
from accelerate import PartialState

ASSISTANT_TAG = [14711, 22103]

def main():
    parser = argparse.ArgumentParser(description="SFT on LitBench-Rationales with reasoning+preference supervision only.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base instruct model to fine-tune (≈1B parameters)")
    parser.add_argument("--dataset_name", type=str, default="SAA-Lab/LitBench-Rationales", help="HF dataset name")
    parser.add_argument("--output_dir", type=str, default="./llama_litbench_sft", help="Where to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)  # Reduced for memory efficiency
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--use_flash_attention", type=bool, default=False, help="Whether to use flash attention")
    parser.add_argument("--use_cot", action="store_true", help="Whether to use chain-of-thought reasoning in training")
    args = parser.parse_args()

    # 1. Load tokenizer & model (bfloat16 to reduce memory)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        if "pad_token" not in tokenizer.special_tokens_map:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_length

    # Get device mapping for multi-GPU training
    device_string = PartialState().process_index
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": None,
        "device_map": {"": device_string}  # Assign to correct device for DDP
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))

    # 2. Prepare dataset using appropriate DataLoader based on --use_cot flag
    if args.use_cot:
        print("Using Chain-of-Thought (COT) dataloader with reasoning + preference")
        data_loader = SFTDataLoaderCOT(
            tokenizer=tokenizer,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            max_length=args.max_length
        )
    else:
        print("Using Direct dataloader with preference only (no reasoning)")
        data_loader = SFTDataLoaderDirect(
            tokenizer=tokenizer,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            max_length=args.max_length
        )
    train_ds, val_ds = data_loader.load_datasets()

    response_template_with_context = "\n### Assistant:\n" 
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:] 

    data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=ASSISTANT_TAG,
        tokenizer=tokenizer,
        mlm=False
    )

    # 4. Training configuration
    wandb.login()
    wandb.init(project="litbench_sft")
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        warmup_ratio=0.1,
        bf16=True,
        report_to=["wandb"],
        dataloader_drop_last=True,
        remove_unused_columns=True,
        packing=False,
        eval_packing=False,
        max_seq_length=args.max_length,
        gradient_checkpointing=True,  # Added for memory efficiency
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for DDP
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete – model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
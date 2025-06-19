#!/usr/bin/env python3
"""
LitBench Dataset Rehydration Script

This script rehydrates the LitBench-Test dataset from comment IDs using Reddit API.
It uses the enhanced comment ID dataset (SAA-Lab/LitBench-Test-Release) which 
contains only complete rows with both chosen and rejected comment IDs.

Usage:
    python rehydrate.py [--output_dir DATA_DIR] [--max_workers N]

The script will:
1. Load comment IDs from the enhanced LitBench-Test-Release dataset  
2. Fetch comment data from Reddit API
3. Save rehydrated dataset as 'rehydrated_test_data.csv'
4. Training scripts will automatically use this data if it exists

Requirements:
    - Reddit API credentials (via praw.ini or environment variables)
    - Internet connection
    - Patience (takes ~1-2 hours due to Reddit rate limits)
"""

import pandas as pd
from datasets import load_dataset
from reddit_utils import RedditUtils
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import time
import random
import argparse

# Configuration
DEFAULT_OUTPUT_DIR = "./data"
DEFAULT_MAX_WORKERS = 3  # Conservative for Reddit API
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 2.0

def fetch_comment_data_with_retry(reddit_utils: RedditUtils, comment_id: str, max_retries: int = DEFAULT_MAX_RETRIES):
    """
    Fetches a single comment's data from Reddit with retry logic and rate limiting.
    Returns a dictionary with relevant fields or None on failure.
    """
    if pd.isna(comment_id) or not isinstance(comment_id, str):
        return None
    
    for attempt in range(max_retries):
        try:
            # Add random delay to spread out requests
            delay = DEFAULT_BASE_DELAY + random.uniform(0.1, 0.5)
            time.sleep(delay)
            
            comment_data = reddit_utils.get_comment_by_id(comment_id, include_replies=False)
            if comment_data:
                return {
                    'id': comment_id,
                    'story': comment_data.get('body'),
                    'score': comment_data.get('score'),
                    'author': comment_data.get('author'),
                    'timestamp': comment_data.get('created_utc'),
                    'post_id': comment_data.get('post_id'),
                    'prompt': comment_data.get('post_title'), # The prompt is the post title
                }
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                # Exponential backoff for rate limiting
                wait_time = (2 ** attempt) * DEFAULT_BASE_DELAY + random.uniform(1, 3)
                print(f"Rate limited for comment {comment_id}, waiting {wait_time:.1f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            elif attempt == max_retries - 1:
                # Last attempt failed, give up
                break
            else:
                # Other error, wait a bit and retry
                time.sleep(DEFAULT_BASE_DELAY * (attempt + 1))
                continue
    
    return None

def rehydrate_litbench_test(output_dir: str = DEFAULT_OUTPUT_DIR, max_workers: int = DEFAULT_MAX_WORKERS):
    """
    Rehydrates the LitBench-Test dataset using the enhanced comment ID dataset.
    """
    print("🚀 LITBENCH TEST DATASET REHYDRATION")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path
    output_path = os.path.join(output_dir, "rehydrated_test_data.csv")
    
    # Check if already exists
    if os.path.exists(output_path):
        response = input(f"⚠️  Rehydrated data already exists at {output_path}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Rehydration cancelled.")
            return
    
    # Load the enhanced comment ID dataset (complete rows only)
    print("📊 Loading enhanced comment ID dataset...")
    try:
        id_dataset = load_dataset("SAA-Lab/LitBench-Test-Release")
        id_df = id_dataset['train'].to_pandas()
        print(f"✅ Loaded {len(id_df)} complete rows (100% have both comment IDs)")
    except Exception as e:
        print(f"❌ ERROR: Failed to load enhanced dataset from HuggingFace.")
        print(f"   Error details: {e}")
        return

    # Collect all unique comment IDs to fetch
    print("\n🔍 Collecting unique comment IDs...")
    chosen_ids = id_df['chosen_comment_id'].dropna().unique()
    rejected_ids = id_df['rejected_comment_id'].dropna().unique()
    all_comment_ids = set(chosen_ids) | set(rejected_ids)
    print(f"  > Found {len(all_comment_ids)} unique comment IDs to fetch")

    # Fetch comment data in parallel with rate limiting
    print(f"\n💧 Fetching comments from Reddit using {max_workers} workers...")
    print("⚠️  This takes 1-2 hours due to Reddit API rate limits. Please be patient...")
    
    fetched_comments = {}
    reddit_utils = RedditUtils()  # Initialize once
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(fetch_comment_data_with_retry, reddit_utils, cid): cid for cid in all_comment_ids}
        
        progress = tqdm(as_completed(future_to_id), total=len(all_comment_ids), desc="Fetching comments")
        
        for future in progress:
            comment_id = future_to_id[future]
            try:
                result = future.result()
                if result:
                    fetched_comments[comment_id] = result
            except Exception as e:
                # Log error but continue
                progress.write(f"Error processing comment {comment_id}: {e}")

    print(f"  ✅ Successfully fetched {len(fetched_comments)}/{len(all_comment_ids)} comments")

    # Assemble the final rehydrated dataset
    print("\n🧩 Assembling rehydrated dataset...")
    
    rehydrated_data = []
    
    for _, row in tqdm(id_df.iterrows(), total=len(id_df), desc="Assembling rows"):
        chosen_id = row['chosen_comment_id']
        rejected_id = row['rejected_comment_id']
        
        chosen_comment = fetched_comments.get(chosen_id)
        rejected_comment = fetched_comments.get(rejected_id)
        
        # Determine the prompt - prioritize the chosen story's post title
        prompt = None
        if chosen_comment:
            prompt = chosen_comment.get('prompt')
        elif rejected_comment:
            prompt = rejected_comment.get('prompt')

        # Create rehydrated row with all necessary fields for training
        new_row = {
            'prompt': prompt,
            'chosen_story': chosen_comment.get('story') if chosen_comment else None,
            'rejected_story': rejected_comment.get('story') if rejected_comment else None,
            'chosen_comment_id': chosen_id,
            'rejected_comment_id': rejected_id,
            'chosen_comment_score': chosen_comment.get('score') if chosen_comment else None,
            'rejected_comment_score': rejected_comment.get('score') if rejected_comment else None,
            'chosen_username': chosen_comment.get('author') if chosen_comment else None,
            'rejected_username': rejected_comment.get('author') if rejected_comment else None,
            'chosen_timestamp': chosen_comment.get('timestamp') if chosen_comment else None,
            'rejected_timestamp': rejected_comment.get('timestamp') if rejected_comment else None,
        }
        rehydrated_data.append(new_row)
        
    final_df = pd.DataFrame(rehydrated_data)

    # Calculate completeness statistics
    complete_rows = 0
    for i in range(len(final_df)):
        if pd.notna(final_df.iloc[i]['chosen_story']) and pd.notna(final_df.iloc[i]['rejected_story']):
            complete_rows += 1
    
    completeness_rate = (complete_rows / len(final_df)) * 100

    # Save the rehydrated dataset
    print(f"\n💾 Saving rehydrated dataset to '{output_path}'...")
    final_df.to_csv(output_path, index=False)
    
    print("\n🎉 REHYDRATION COMPLETE! 🎉")
    print(f"   📊 Dataset saved: {len(final_df)} rows")
    print(f"   📈 Complete rows: {complete_rows}/{len(final_df)} ({completeness_rate:.1f}%)")
    print(f"   📍 Location: {output_path}")
    print("\n✨ Training scripts will now automatically use this rehydrated data!")
    print("   Run your training commands as usual - no additional setup needed.")

def main():
    parser = argparse.ArgumentParser(description='Rehydrate LitBench-Test dataset from comment IDs')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save rehydrated data (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--max_workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'Number of worker threads for fetching (default: {DEFAULT_MAX_WORKERS})')
    
    args = parser.parse_args()
    
    print("LitBench Dataset Rehydration")
    print("============================")
    print(f"Output directory: {args.output_dir}")
    print(f"Max workers: {args.max_workers}")
    print("")
    
    rehydrate_litbench_test(args.output_dir, args.max_workers)

if __name__ == "__main__":
    main() 
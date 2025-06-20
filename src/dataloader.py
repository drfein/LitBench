import os
from datasets import load_dataset, Dataset
from typing import Dict, Any, Optional, List, Tuple
import torch
import random
from textwrap import dedent
import pandas as pd

class LitBenchDataLoader:
    """
    DataLoader for the LitBench datasets, supporting both training and testing splits.
    Ensures compatibility with TRL's RewardTrainer by providing the necessary fields:
    - input_ids_chosen
    - attention_mask_chosen
    - input_ids_rejected
    - attention_mask_rejected
    """
    
    def __init__(
        self, 
        tokenizer,
        max_length: int = 2048,
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        use_fast_tokenizer: bool = True
    ):
        """
        Initialize the LitBench dataloader.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache the datasets
            data_dir: Directory to store processed datasets
            use_fast_tokenizer: Whether to use fast tokenizer implementation
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.data_dir = data_dir or "./data"
        self.use_fast_tokenizer = use_fast_tokenizer
        
        # Create directories if they don't exist
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        if self.data_dir:
            os.makedirs(self.data_dir, exist_ok=True)
            
        # Process dataset names
        self.train_dataset_name = "SAA-Lab/LitBench-Train"
    
    def load_datasets(self) -> Tuple[Any, Any, Any]:
        """
        Load train, validation, and test datasets.
        Uses rehydrated test data (required) and HuggingFace train data.
        
        Returns:
            Tuple containing (train_dataset, val_dataset, test_dataset)
        """
        rehydrated_test_path = os.path.join(self.data_dir, "rehydrated_test_data.csv")
        
        if not os.path.exists(rehydrated_test_path):
            raise FileNotFoundError(
                f"❌ Rehydrated test data not found at {rehydrated_test_path}\n"
                f"   Please run 'python scripts/rehydrate.py' first to create the rehydrated dataset.\n"
                f"   This repository now uses only rehydrated data for testing."
            )
        
        print(f"🎉 Loading rehydrated test data from {rehydrated_test_path}")
        
        # Load rehydrated test data
        try:
            rehydrated_df = pd.read_csv(rehydrated_test_path)
            test_dataset = Dataset.from_pandas(rehydrated_df)
            print(f"✅ Loaded {len(test_dataset)} rows from rehydrated test data")
            
            # Verify required fields exist
            required_fields = ["prompt", "chosen_story", "rejected_story"]
            missing_fields = [field for field in required_fields if field not in test_dataset.column_names]
            if missing_fields:
                raise ValueError(f"Missing required fields in rehydrated data: {missing_fields}")
                
        except Exception as e:
            raise RuntimeError(f"❌ Error loading rehydrated data: {e}")
        
        # Always load train dataset from HuggingFace (no rehydration needed for train)
        print(f"Loading train dataset from {self.train_dataset_name}...")
        train_dataset = load_dataset(
            self.train_dataset_name, 
            split="train", 
            cache_dir=self.cache_dir
        )
        
        # Create a validation subset from the test dataset
        val_subset_size = min(500, len(test_dataset))
        val_subset_indices = random.sample(range(len(test_dataset)), val_subset_size)
        val_dataset = test_dataset.select(val_subset_indices)
        
        print(f"Dataset sizes: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Confirm the datasets have the required fields for RewardTrainer
        required_fields = ["prompt", "chosen_story", "rejected_story"]
        
        for field in required_fields:
            if field not in train_dataset.column_names:
                raise ValueError(f"Missing required field '{field}' in train dataset")
            if field not in test_dataset.column_names:
                raise ValueError(f"Missing required field '{field}' in test dataset")
                
        return train_dataset, val_dataset, test_dataset
    
    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, List]:
        """
        Preprocess and tokenize the examples for BT reward modeling.
        Creates the specific fields required by TRL's RewardTrainer.
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            Dictionary containing tokenized inputs for chosen and rejected examples
        """
        chosen_texts = examples["chosen_story"]
        rejected_texts = examples["rejected_story"]
        
        tokenized_chosen = self.tokenizer(
            chosen_texts, 
            padding="max_length",
            truncation=True, 
            max_length=self.max_length,
            return_tensors=None
        )
        
        tokenized_rejected = self.tokenizer(
            rejected_texts, 
            padding="max_length",
            truncation=True, 
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }
    
    def prepare_datasets(self, model_name_safe: str) -> Tuple[Any, Any, Any]:
        """
        Load and preprocess all datasets.
        
        Args:
            model_name_safe: Safe version of model name for cache files
            
        Returns:
            Tuple containing (processed_train_dataset, processed_val_dataset, processed_test_dataset)
        """
        train_dataset, val_dataset, test_dataset = self.load_datasets()
        
        processed_data_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Generate cache file names
        cache_file_prefix = f"processed_{model_name_safe}_litbench_{self.max_length}"
        
        print(f"Tokenizing train dataset...")
        processed_train = train_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=16,
            desc="Tokenizing train dataset",
            num_proc=1,
            load_from_cache_file=True,
            cache_file_name=os.path.join(processed_data_dir, f"{cache_file_prefix}_train.arrow"),
            remove_columns=train_dataset.column_names  # Remove original columns to keep only the model inputs
        )
        
        print(f"Tokenizing validation dataset...")
        processed_val = val_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=16,
            desc="Tokenizing validation dataset",
            num_proc=1,
            load_from_cache_file=True,
            cache_file_name=os.path.join(processed_data_dir, f"{cache_file_prefix}_val.arrow"),
            remove_columns=val_dataset.column_names  # Remove original columns to keep only the model inputs
        )
        
        print(f"Tokenizing test dataset...")
        processed_test = test_dataset.map(
            self.preprocess_function,
            batched=True,
            batch_size=16,
            desc="Tokenizing test dataset",
            num_proc=1,
            load_from_cache_file=True,
            cache_file_name=os.path.join(processed_data_dir, f"{cache_file_prefix}_test.arrow"),
            remove_columns=test_dataset.column_names  # Remove original columns to keep only the model inputs
        )
        
        # Verify that the processed datasets have the required fields for TRL's RewardTrainer
        required_fields = ["input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"]
        
        for field in required_fields:
            for dataset_name, dataset in [("train", processed_train), ("val", processed_val), ("test", processed_test)]:
                if field not in dataset.column_names:
                    raise ValueError(f"Missing required field '{field}' in processed {dataset_name} dataset")
        
        print(f"Datasets processed: Train: {len(processed_train)}, Val: {len(processed_val)}, Test: {len(processed_test)}")
        print(f"Dataset fields: {processed_train.column_names}")
        
        return processed_train, processed_val, processed_test

class SFTDataLoaderCOT:
    """
    DataLoader for instruction tuning on LitBench-Rationales dataset.
    Formats examples for supervised fine-tuning with completion-only loss.
    """
    
    def __init__(
        self, 
        tokenizer,
        model_name: str,
        dataset_name: str = "SAA-Lab/LitBench-Rationales",
        max_length: int = 2048,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the SFT dataloader.
        
        Args:
            tokenizer: HuggingFace tokenizer
            model_name: Name of the model being used (determines prompt format)
            dataset_name: HuggingFace dataset name
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache the datasets
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Define prompt templates based on model architecture
        self.QWEN_PROMPT = """You are evaluating two creative writing responses (A and B) to the same writing prompt. These responses are similar to those posted on Reddit writing subreddits like r/WritingPrompts.

Your task is to predict which response would receive more upvotes from the Reddit community. Reddit users typically upvote creative writing that is engaging, original, well-written, and emotionally resonant.

When making your prediction, consider what makes content popular on Reddit:
- Originality and uniqueness of ideas
- Engaging narrative style and pacing
- Emotional impact and relatability
- Clever twists or satisfying conclusions
- Technical quality of writing

Story A:
{story_a}

Story B:
{story_b}

This is an experiment to test how well language models can predict human preferences in creative writing as expressed through Reddit's voting system.

Your verdict MUST follow this exact format:
Reasoning: [explain which response would likely get more Reddit upvotes and why]
Preferred: [A or B] (the one you predict would get more upvotes)
"""

        self.LLAMA_PROMPT = """Evaluate creative writing responses A and B.

Story A:
{story_a}

Story B:
{story_b}

Consider these aspects:
- Originality: unique concepts, unexpected elements
- Imagery: sensory language and descriptions
- Emotional impact: how the writing affects the reader
- Coherence: logical flow and narrative structure
- Technical skill: language use and style

FORMAT REQUIRED:
Reasoning: [your evaluation]
Preferred: [A or B]
"""

        # Define custom response marker
        self.response_marker = "### Assistant:"

    def _extract_assistant_tag(self) -> List[int]:
        """
        Return the custom response marker as token IDs for use in DataCollatorForCompletionOnlyLM.
        """
        return self.tokenizer.encode(self.response_marker, add_special_tokens=False)

    def _format_user_prompt(self, ex: dict) -> str:
        """
        Fill the prompt template that will appear in the user turn.
        """
        if "llama" in self.model_name.lower():
            tpl = self.LLAMA_PROMPT
        else:
            tpl = self.QWEN_PROMPT

        return tpl.format(
            story_a=ex["story_a"],
            story_b=ex["story_b"],
        ).strip()

    def _format_assistant_answer(self, ex: dict) -> str:
        """
        Build the assistant completion: chain-of-thought (Reasoning) + decision (Preferred).
        """
        rationale = ex.get("rationale", "").strip()
        if not rationale.lower().startswith("reasoning:"):
            rationale = f"Reasoning: {rationale}"

        label = ex.get("preferred", ex.get("chosen_story", "")).strip()
        if label not in {"A", "B"}:
            raise ValueError(f"Label must be 'A' or 'B', got: {label}")

        return f"{rationale}\nPreferred: {label}"

    def build_chat_example(self, example: dict) -> dict:
        """
        Format dataset example as a single text string with custom response marker.
        """
        user_prompt = self._format_user_prompt(example)
        assistant_answer = self._format_assistant_answer(example)
        # Combine with custom response marker
        chat_text = f"{user_prompt}\n\n{self.response_marker}\n{assistant_answer}"
        return {"text": chat_text, "response_marker": self.response_marker}

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Load and prepare train and validation datasets.
        
        Returns:
            Tuple containing (train_dataset, val_dataset)
        """
        print(f"Loading dataset from {self.dataset_name}...")
        raw_dataset = load_dataset(self.dataset_name, split="train", cache_dir=self.cache_dir)
        
        # Quick 95/5 split for validation
        raw_split = raw_dataset.train_test_split(test_size=0.05, seed=42)
        train_raw = raw_split["train"]
        val_raw = raw_split["test"]
        
        print(f"Dataset sizes: Train: {len(train_raw)}, Val: {len(val_raw)}")
        
        # Map to chat format texts
        train_ds = train_raw.map(
            lambda ex: self.build_chat_example(ex),
            remove_columns=train_raw.column_names
        )
        val_ds = val_raw.map(
            lambda ex: self.build_chat_example(ex),
            remove_columns=val_raw.column_names
        )
        
        return train_ds, val_ds
    
class SFTDataLoaderDirect:
    """
    DataLoader for instruction tuning on LitBench-Rationales dataset.
    Formats examples for supervised fine-tuning with completion-only loss.
    """
    
    def __init__(
        self, 
        tokenizer,
        model_name: str,
        dataset_name: str = "SAA-Lab/LitBench-Rationales",
        max_length: int = 2048,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the SFT dataloader.
        
        Args:
            tokenizer: HuggingFace tokenizer
            model_name: Name of the model being used (determines prompt format)
            dataset_name: HuggingFace dataset name
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache the datasets
        """
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Define prompt templates based on model architecture
        self.QWEN_PROMPT = """You are evaluating two creative writing responses (A and B) to the same writing prompt. These responses are similar to those posted on Reddit writing subreddits like r/WritingPrompts.

Your task is to predict which response would receive more upvotes from the Reddit community. Reddit users typically upvote creative writing that is engaging, original, well-written, and emotionally resonant.

When making your prediction, consider what makes content popular on Reddit:
- Originality and uniqueness of ideas
- Engaging narrative style and pacing
- Emotional impact and relatability
- Clever twists or satisfying conclusions
- Technical quality of writing

Story A:
{story_a}

Story B:
{story_b}

This is an experiment to test how well language models can predict human preferences in creative writing as expressed through Reddit's voting system.

Your verdict MUST follow this exact format:
Preferred: [A or B] (the one you predict would get more upvotes)
"""

        self.LLAMA_PROMPT = """Evaluate creative writing responses A and B.

Story A:
{story_a}

Story B:
{story_b}

Consider these aspects:
- Originality: unique concepts, unexpected elements
- Imagery: sensory language and descriptions
- Emotional impact: how the writing affects the reader
- Coherence: logical flow and narrative structure
- Technical skill: language use and style

FORMAT REQUIRED:
Preferred: [A or B]
"""

        # Define custom response marker
        self.response_marker = "### Assistant:"

    def _extract_assistant_tag(self) -> List[int]:
        """
        Return the custom response marker as token IDs for use in DataCollatorForCompletionOnlyLM.
        """
        return self.tokenizer.encode(self.response_marker, add_special_tokens=False)

    def _format_user_prompt(self, ex: dict) -> str:
        """
        Fill the prompt template that will appear in the user turn.
        """
        if "llama" in self.model_name.lower():
            tpl = self.LLAMA_PROMPT
        else:
            tpl = self.QWEN_PROMPT

        return tpl.format(
            story_a=ex["story_a"],
            story_b=ex["story_b"],
        ).strip()

    def _format_assistant_answer(self, ex: dict) -> str:
        """
        Build the assistant completion: just the decision (Preferred).
        """
        label = ex.get("preferred", ex.get("chosen_story", "")).strip()
        if label not in {"A", "B"}:
            raise ValueError(f"Label must be 'A' or 'B', got: {label}")

        return f"Preferred: {label}"

    def build_chat_example(self, example: dict) -> dict:
        """
        Format dataset example as a single text string with custom response marker.
        """
        user_prompt = self._format_user_prompt(example)
        assistant_answer = self._format_assistant_answer(example)
        # Combine with custom response marker
        chat_text = f"{user_prompt}\n\n{self.response_marker}\n{assistant_answer}"
        return {"text": chat_text, "response_marker": self.response_marker}

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Load and prepare train and validation datasets.
        
        Returns:
            Tuple containing (train_dataset, val_dataset)
        """
        print(f"Loading dataset from {self.dataset_name}...")
        raw_dataset = load_dataset(self.dataset_name, split="train", cache_dir=self.cache_dir)
        
        # Quick 95/5 split for validation
        raw_split = raw_dataset.train_test_split(test_size=0.05, seed=42)
        train_raw = raw_split["train"]
        val_raw = raw_split["test"]
        
        print(f"Dataset sizes: Train: {len(train_raw)}, Val: {len(val_raw)}")
        
        # Map to chat format texts
        train_ds = train_raw.map(
            lambda ex: self.build_chat_example(ex),
            remove_columns=train_raw.column_names
        )
        val_ds = val_raw.map(
            lambda ex: self.build_chat_example(ex),
            remove_columns=val_raw.column_names
        )
        
        return train_ds, val_ds

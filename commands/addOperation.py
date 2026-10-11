# commands/addOperation.py

import os
import pandas as pd
import torch
import pickle
from datasets import Dataset, DatasetDict

from coreComponents.transformers_plugin import TransformersPlugin
from coreComponents.tokenization import TrajectoryTokenizer
from coreComponents.dataCollator import TrajectoryDataCollator
from coreComponents.trajectory_operations_plugin import TrajectoryOperationsPlugin
from coreComponents.offline_training import train_operation_model
from coreComponents.partitioning import PartitioningModule
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def add_operation(base_path,
                 operation_name,
                 transformer_name,
                 training_data,
                 args,
                 operation_script):

    """
    Add a new operation:
        - validate transformer exists
        - check if operation already has this transformer
        - tokenize dataset (operation-dependent)
        - train model
        - save model using partitioning module
        - ONLY THEN update operation registry
    """

    # ----------------------------------------------------------
    # 1. Validate transformer exists
    # ----------------------------------------------------------
    PT = PartitioningModule(base_path)
    tplugin = TransformersPlugin(base_path)
    build_model_fn = tplugin.load_model_builder(transformer_name)

    if build_model_fn is None:
        logger.error(f"Transformer '{transformer_name}' not found.")
        raise ValueError(f"Transformer '{transformer_name}' not found.")

    # ----------------------------------------------------------
    # 2. Check if operation already has this transformer
    # ----------------------------------------------------------
    oplugin = TrajectoryOperationsPlugin(base_path)
    # Before any training we need to make sure the operation script is valid
    oplugin.validate_operation_script(operation_script, operation_name)
    # Function 1: Check if transformer already exists for this operation
    if oplugin.has_transformer(operation_name, transformer_name):
        logger.error(
            f"Operation '{operation_name}' already has transformer "
            f"'{transformer_name}' registered."
        )
        logger.error(
            "Training aborted. Use a different transformer name or operation name."
        )
        return

    # ----------------------------------------------------------
    # 3. Validate operation script exists
    # ----------------------------------------------------------
    script_path = os.path.abspath(operation_script)
    if not os.path.exists(script_path):
        logger.error(f"Operation script not found at path: {script_path}")
        raise FileNotFoundError(f"Operation script not found: {script_path}")

    # ----------------------------------------------------------
    # 4. Load training CSV
    # ----------------------------------------------------------
    df = pd.read_csv(training_data)
    logger.info(f"Loaded training data with {len(df)} rows")

    # ----------------------------------------------------------
    # 5. NEW: Process CSV based on operation type
    # ----------------------------------------------------------
    tokenizer = TrajectoryTokenizer(resolution=args.get("resolution", 10))
    
    # Determine which columns to tokenize based on operation type
    operation_type = args["operation"]
    
    if operation_type == "summarization":
        columns_to_tokenize = ["trajectory", "summary"]
    elif operation_type == "generation":
        columns_to_tokenize = ["trajectory"]
    elif operation_type == "classification":
        columns_to_tokenize = ["trajectory"]  # labels are already numeric
    elif operation_type == "next_point":
        columns_to_tokenize = ["trajectory"]
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")
    
    # Process CSV: Convert GPS strings to H3 tokens
    logger.info(f"Converting GPS to H3 tokens for columns: {columns_to_tokenize}")
    df_processed = TrajectoryTokenizer.process_csv(
        df=df,
        columns_to_tokenize=columns_to_tokenize,
        resolution=tokenizer.resolution
    )
    print(df_processed)
    # Build vocabulary from processed data
    logger.info("Building vocabulary...")
    all_h3_strings = []
    for col in columns_to_tokenize:
        if col in df_processed.columns:
            col_h3_strings = df_processed[col].dropna().tolist()
            all_h3_strings.extend(col_h3_strings)
    
    tokenizer.build_vocab(all_h3_strings)
    logger.info(f"Vocabulary built with {tokenizer.get_vocab_size()} tokens")
    
    # ----------------------------------------------------------
    # 6. NEW: Prepare dataset for the specific operation
    # ----------------------------------------------------------
    logger.info(f"Preparing dataset for {operation_type} operation...")
    
    if operation_type == "summarization":
        dataset = prepare_summarization_dataset(df_processed, tokenizer)
    elif operation_type == "generation":
        dataset = prepare_generation_dataset(df_processed, tokenizer)
    elif operation_type == "classification":
        dataset = prepare_classification_dataset(df_processed, tokenizer)
    elif operation_type == "next_point":
        dataset = prepare_next_point_dataset(df_processed, tokenizer)
    else:
        raise ValueError(f"Unsupported operation type: {operation_type}")
    # ----------------------------------------------------------
    # 6.5. Save processed H3 dataset to CSV
    # ----------------------------------------------------------
    
    saved_tokenized_data_path = PT.save_tokenized_dataset(tokenized_dataset=df_processed,
        operation_name=operation_name,
        original_data_path=training_data)
    logging.info(f"Tokenized Data saved to: {saved_tokenized_data_path}")
    # ----------------------------------------------------------
    # 7. Create data collator based on operation type
    # ----------------------------------------------------------
    collator = create_data_collator(tokenizer, operation_type, args.get("max_length", 256))
    
    # ----------------------------------------------------------
    # 8. Train model (updated training function)
    # ----------------------------------------------------------
    logger.info(f"Training {transformer_name} model for {operation_name}...")
    model = train_operation_model(
        build_model_fn,
        tokenizer,
        dataset,
        collator,
        args,
    )
    # model = train_operation_model(
    #     build_model_fn=build_model_fn,
    #     tokenizer=tokenizer,
    #     dataset=dataset,
    #     data_collator=collator,
    #     training_args=args,
    # )
    logger.info(f"New model trained successfully")
    
    # ----------------------------------------------------------
    # 9. Save model using Partitioning Module
    # ----------------------------------------------------------
    model_name = f"{operation_name}_{transformer_name}"
    
    model_path = PT.save_model(
        model=model,
        operation_name=operation_name,
        model_name=model_name,
        training_trajectories_path=training_data
    )
    logger.info(f"Model saved to: {model_path}")
    
    # ----------------------------------------------------------
    # 10. Save metadata
    # ----------------------------------------------------------
    model_info = {
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer,
        "transformer": transformer_name,
        "operation_name": operation_name,
        "model_name": model_name,
        "training_data_path": training_data,
        "resolution": tokenizer.resolution,
        "vocab_size": tokenizer.get_vocab_size(),
    }
    
    metadata_path = os.path.join(os.path.dirname(model_path), f"{model_name}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(model_info, f)
    logger.info(f"Model metadata saved")
    
    # ----------------------------------------------------------
    # 11. Update operations registry
    # ----------------------------------------------------------
    oplugin.register(operation_name, script_path, transformer_name, model_path)
    logger.info(f"Operation '{operation_name}' registered successfully")

    return model_path


# ----------------------------------------------------------
# Helper functions for dataset preparation
# ----------------------------------------------------------

def prepare_summarization_dataset(df, tokenizer):
    """
    Input: original trajectory
    Output: summary trajectory
    """

    def map_fn(row):
        inp = "summarize: " + row["trajectory"]
        out = row["summary"]

        return {
            "input_ids": tokenizer.encode(inp),
            "labels": tokenizer.encode(out),
        }

    ds = Dataset.from_pandas(df[["trajectory", "summary"]])
    ds = ds.map(map_fn, remove_columns=["trajectory", "summary"])

    return DatasetDict({
        "train": ds,
        "validation": ds,
    })


def prepare_generation_dataset(df, tokenizer):
    """Prepare dataset for generation task (auto-regressive)."""
    data = []
    for idx, row in df.iterrows():
        if pd.isna(row["trajectory"]):
            continue
        
        # For generation, input and labels are the same
        input_ids = tokenizer.encode(row["trajectory"])
        
        data.append({
            "input_ids": input_ids,
            "labels": input_ids.copy()  # Same as input for LM training
        })
    
    logger.info(f"Prepared {len(data)} samples for generation")
    return data


def prepare_classification_dataset(df, tokenizer):
    """Prepare dataset for classification task."""
    # Create label mapping
    labels = sorted(df["label"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    data = []
    for idx, row in df.iterrows():
        if pd.isna(row["trajectory"]) or pd.isna(row["label"]):
            continue
        
        input_ids = tokenizer.encode(row["trajectory"])
        label = label_map[row["label"]]
        
        data.append({
            "input_ids": input_ids,
            "labels": label  # Single integer label
        })
    
    logger.info(f"Prepared {len(data)} samples for classification with {len(labels)} classes")
    return data


def prepare_next_point_dataset(df, tokenizer):
    """Prepare dataset for next point prediction."""
    data = []
    for idx, row in df.iterrows():
        if pd.isna(row["trajectory"]):
            continue
        
        tokens = row["trajectory"].split()
        if len(tokens) < 2:
            continue
        
        # Input: all tokens except last
        input_text = " ".join(tokens[:-1])
        # Label: last token only
        label_text = tokens[-1]
        
        input_ids = tokenizer.encode(input_text)
        label_ids = tokenizer.encode(label_text)
        
        data.append({
            "input_ids": input_ids,
            "labels": label_ids
        })
    
    logger.info(f"Prepared {len(data)} samples for next point prediction")
    ds = Dataset.from_list(data)
    return DatasetDict({
        "train": ds,
        "validation": ds,
    })


def create_data_collator(tokenizer, operation_type, max_length=256):
    """Create appropriate data collator for the operation type."""
    from coreComponents.dataCollator import (
        TrajectoryDataCollator, 
        SummarizationCollator, 
        GenerationCollator
    )
    
    if operation_type == "summarization":
        return SummarizationCollator(tokenizer, max_length)
    elif operation_type == "generation":
        return GenerationCollator(tokenizer, max_length)
    else:
        return TrajectoryDataCollator(tokenizer, max_length, operation_type)
#!/usr/bin/env python3
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from slora.trainers.slora_trainer import SLoRATrainer
from slora.utils.seed import set_seed
from slora.utils.logging import setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: Dict[str, Any], tokenizer):
    """Load and prepare dataset."""
    dataset = load_dataset(
        config["data"]["dataset_name"],
        split=config["data"]["train_split"],
    )
    eval_dataset = load_dataset(
        config["data"]["dataset_name"],
        split=config["data"]["eval_split"],
    )

    def formatting_func(example):
        """Format example as chat template."""
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}
        except Exception:
            return {"text": None}

    dataset_orig_size = len(dataset)
    dataset = dataset.map(formatting_func)
    dataset = dataset.filter(lambda x: x["text"] is not None)
    logger.info(f"Train dataset: kept {len(dataset)}/{dataset_orig_size} examples (skipped {dataset_orig_size - len(dataset)})")

    eval_orig_size = len(eval_dataset)
    eval_dataset = eval_dataset.map(formatting_func)
    eval_dataset = eval_dataset.filter(lambda x: x["text"] is not None)
    logger.info(f"Eval dataset: kept {len(eval_dataset)}/{eval_orig_size} examples (skipped {eval_orig_size - len(eval_dataset)})")

    def tokenize_func(examples):
        """Tokenize text."""
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["data"]["max_seq_length"],
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=(
            list(dataset.column_names)
            if isinstance(dataset.column_names, dict)
            else dataset.column_names
        ),
    )
    eval_dataset = eval_dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=(
            list(eval_dataset.column_names)
            if isinstance(eval_dataset.column_names, dict)
            else eval_dataset.column_names
        ),
    )

    return dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="Train SLoRA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging()

    set_seed(config["training"]["seed"])

    logger.info(f"Loading model: {config['model']['name']}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = {}
    if config["model"].get("use_flash_attention_2", False):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    if config["model"].get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, config["model"].get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_quant_type=config["model"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=config["model"].get(
                "bnb_4bit_use_double_quant", True
            ),
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
        **model_kwargs,
    )

    if config["model"].get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading and preparing dataset")
    train_dataset, eval_dataset = prepare_data(config, tokenizer)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        max_steps=config["training"].get("max_steps", -1),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"].get(
            "per_device_eval_batch_size",
            config["training"]["per_device_train_batch_size"],
        ),
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        learning_rate=config["training"]["learning_rate"],
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        bf16=config["training"]["bf16"],
        tf32=config["training"].get("tf32", True),
        max_grad_norm=config["training"]["max_grad_norm"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        eval_strategy=config["training"]["evaluation_strategy"],
        eval_steps=config["training"]["eval_steps"],
        load_best_model_at_end=config["training"]["load_best_model_at_end"],
        seed=config["training"]["seed"],
        dataloader_num_workers=config["training"]["dataloader_num_workers"],
        dataloader_pin_memory=config["training"]["dataloader_pin_memory"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        report_to=config["logging"]["report_to"],
        run_name=config["logging"].get("wandb_run_name", None),
    )

    gate_config = None
    enable_gate = config["slora"].get("enable", True)
    if enable_gate:
        gate_config = {
            "m": config["slora"]["m"],
            "k": config["slora"]["k"],
            "tau_n": config["slora"]["tau_n"],
            "burn_in": config["slora"]["burn_in"],
            "seed": config["slora"]["seed"],
            "reorth_every": config["slora"]["reorth_every"],
        }

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, pad_to_multiple_of=8
    )

    trainer = SLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        gate_config=gate_config,
        enable_gate=enable_gate,
    )

    logger.info("Starting training")
    trainer.train()

    logger.info("Saving final model")
    trainer.save_model()
    tokenizer.save_pretrained(config["training"]["output_dir"])

    if enable_gate and trainer.gate is not None:
        logger.info(f"Final acceptance rate: {trainer.gate.acceptance_rate():.4f}")
        logger.info(
            f"Total accepted steps: {trainer.gate.accepted_count}/{trainer.gate.step_count}"
        )


if __name__ == "__main__":
    main()

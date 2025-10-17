from typing import Dict, Any
from datasets import load_dataset


def prepare_data(config: Dict[str, Any], tokenizer, logger):
    dataset = load_dataset(
        config["data"]["dataset_name"],
        split=config["data"]["train_split"],
    )
    eval_dataset = load_dataset(
        config["data"]["dataset_name"],
        split=config["data"]["eval_split"],
    )

    def formatting_func(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}
        except Exception:
            return {"text": None}

    dataset_orig_size = len(dataset)  # type: ignore
    dataset = dataset.map(formatting_func)
    dataset = dataset.filter(lambda x: x["text"] is not None)
    logger.info(
        f"Train dataset: kept {len(dataset)}/{dataset_orig_size} examples"  # type: ignore
    )

    eval_orig_size = len(eval_dataset)  # type: ignore
    eval_dataset = eval_dataset.map(formatting_func)
    eval_dataset = eval_dataset.filter(lambda x: x["text"] is not None)
    logger.info(
        f"Eval dataset: kept {len(eval_dataset)}/{eval_orig_size} examples"  # type: ignore
    )

    def tokenize_func(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["data"]["max_seq_length"],
            padding="max_length",
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

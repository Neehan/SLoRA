def format_tulu3(example):
    messages = example["messages"]
    formatted = "<bos>"
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
        elif msg["role"] == "assistant" or msg["role"] == "model":
            formatted += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
        else:
            return None
    if formatted == "<bos>":
        return None
    return formatted


def format_open_platypus(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    formatted = f"<bos><start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>\n"
    return formatted


def format_gsm8k(example):
    question = example["question"]
    answer = example["answer"]
    formatted = f"<bos><start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>\n"
    return formatted

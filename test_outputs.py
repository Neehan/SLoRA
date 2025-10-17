import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16, device_map="cpu")

input_ids = torch.randint(0, 1000, (1, 10))
attention_mask = torch.ones_like(input_ids)

# Test return_dict=True
outputs_dict = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
print("return_dict=True:")
print(f"  logits shape: {outputs_dict.logits.shape}")
print(f"  hidden_states is tuple: {isinstance(outputs_dict.hidden_states, tuple)}")
print(f"  num hidden layers: {len(outputs_dict.hidden_states)}")
print(f"  last hidden shape: {outputs_dict.hidden_states[-1].shape}")

# Test return_dict=False
outputs_tuple = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=False)
print("\nreturn_dict=False:")
print(f"  outputs is tuple: {isinstance(outputs_tuple, tuple)}")
print(f"  len(outputs): {len(outputs_tuple)}")
print(f"  outputs[0] shape (logits): {outputs_tuple[0].shape}")
print(f"  outputs[1] is tuple: {isinstance(outputs_tuple[1], tuple)}")
if isinstance(outputs_tuple[1], tuple):
    print(f"  len(outputs[1]): {len(outputs_tuple[1])}")
    print(f"  outputs[1][-1] shape (last hidden): {outputs_tuple[1][-1].shape}")

# Verify they're the same
print("\nVerification:")
print(f"  logits match: {torch.allclose(outputs_dict.logits, outputs_tuple[0], atol=1e-5)}")
print(f"  hidden[-1] match: {torch.allclose(outputs_dict.hidden_states[-1], outputs_tuple[1][-1], atol=1e-5)}")

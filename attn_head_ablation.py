import os
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.modeling_qwen import (
    enable_monkey_patched_qwen,
    clean_property
)
from tqdm import tqdm
import pandas as pd

# -----------------------------
# load model

model_name = "/data/bowen/models/dpsk_qwen2.5_distill_7B"  # 或你的本地路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda")
# -----------------------------
# -----------------------------
# monkey patch
enable_monkey_patched_qwen(model)
# -----------------------------

probe_weight = torch.load('models/difficulty_vector_dpsk_qwen.pth').squeeze()  # [3584]

difficulty_head_config = {
    "difficulty_vector": probe_weight,  # [3584]
    "all_outputs": []
}

# inject attribution into each layer's self_attn
for layer in model.model.layers:
    setattr(layer.self_attn, "difficulty_head", copy.deepcopy(difficulty_head_config))

# Verify whether the injection is successful
# print("Verifying injection:")
# for i, layer in enumerate(model.model.layers):
#     has_attr = hasattr(layer.self_attn, 'difficulty_head')
#     print(f"  Layer {i}: {'✓' if has_attr else '✗'}")

dfd = pd.read_parquet("data/test_data/attention_head_test.parquet")

log = {}

for difficulty in sorted(dfd['difficulty'].unique()):
    all_layer_head_scores = []
    the_dfd = dfd[dfd['difficulty'] == difficulty]
    for question in tqdm(the_dfd['templated_question']):
        dict_question = question.tolist()[0]
        messages = [
            {'content': 'After solving the mathematical problem, place the final answer inside \\boxed{}', 'role': 'system'},
            dict_question
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)

    for layer_idx in range(len(model.model.layers)):
        attn_module = model.model.layers[layer_idx].self_attn
        if hasattr(attn_module, "difficulty_head"):
            scores = attn_module.difficulty_head["all_outputs"]
            all_layer_head_scores.append(scores)
        else:
            all_layer_head_scores.append([0.0] * 28)
            print(f"×××: Layer {layer_idx}: No difficulty_head found, defaulting to zeros.")

    # average
    log[difficulty.item()] = torch.tensor(all_layer_head_scores) / len(the_dfd['templated_question'])  # [28_layers, 28_heads]
    print(f'>>> Difficulty: {difficulty}, saving results...')
    torch.save(log, "data/results/dpsk_qwen_difficulty_head_scores.pth")

    # clear
    for layer_idx in range(len(model.model.layers)):
        clean_property(model, f"layers.{layer_idx}.self_attn", "difficulty_head")
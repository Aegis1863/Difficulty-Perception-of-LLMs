'''
Specially test the difficulty identification of data before and after modifying attention head
'''

import os
import copy
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.modeling_qwen import enable_monkey_patched_qwen_last
import warnings
import torch
import torch.nn as nn
from ast import literal_eval

warnings.filterwarnings('ignore')

class RegressionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    linear_model = torch.load("models/difficulty_probe_qwen2.5.pth", weights_only=False)
    linear_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda")

    if int(args.weight_head) == 1:
        enable_monkey_patched_qwen_last(model)
        weight_head_config = {
            "simple_weight": args.scale_simple,
            "difficulty_weight": args.scale_difficult,
        }
        for layer in model.model.layers:  # only the last attention module is changed
            setattr(layer.self_attn, "weight_head", copy.deepcopy(weight_head_config))
        print('------------------------------')
        print('>>> Attention module changed.')
        print('------------------------------')

    model.eval()


    dfd_sampled = pd.read_csv(args.data)

    # Check if the file exists
    if os.path.exists(args.save_path):
        log = pd.read_csv(args.save_path)
        for col, default in [
            ("pred_difficulty", None),
            ("real_dfficulty", None),
        ]:
            if col not in log.columns:
                log[col] = [default] * len(log)
    else:
        log = pd.DataFrame({
            "pred_difficulty": [None] * len(dfd_sampled),
            "real_difficulty": [None] * len(dfd_sampled),
        })

    for idx, row in tqdm(dfd_sampled.iterrows(), total=len(dfd_sampled)):
        if pd.notna(log.at[idx, "pred_difficulty"]):
            continue

        try:
            chat = literal_eval(row["templated_question"])
            chat.insert(0, {"role": "system", "content": "After solving the mathematical problem, place the final answer inside \\boxed{}"})

            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=4096).to('cuda')

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
            last_token_emb = last_hidden[:, -1, :].squeeze(0)

            # saving
            log.at[idx, "pred_difficulty"] = linear_model(last_token_emb).item()
            log.at[idx, "real_difficulty"] = row['difficulty']
            log.to_csv(args.save_path, index=False)

        except Exception as e:
            print(f"[Error] index {idx} failed: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate difficulty")
    parser.add_argument("--device", type=int, default=1, help="GPU number")
    parser.add_argument("--model", type=str, default='/data/bowen/models/qwen2.5-7B-instruct', help="HuggingFace model name or local path")
    parser.add_argument("--save_path", type=str, default='data/results/weight_qwen_difficulty_deepmath.csv', help="Result saved to CSV file path")
    parser.add_argument("--data", type=str, default="data/test_data/deepmath_sampled.csv", help="Benchmark path")
    parser.add_argument("--weight_head", type=int, default=1, help="Whether to apply weight scaling (the next two parameters are invalid if not 1)")
    parser.add_argument("--scale_simple", type=float, default=0.1, help="Simple identification head scaling ratio")
    parser.add_argument("--scale_difficult", type=float, default=2, help="Difficult identification head scaling ratio")
    args = parser.parse_args()
    main(args)
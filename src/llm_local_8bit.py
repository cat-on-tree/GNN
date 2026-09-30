import os
import json
import torch
import math
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# === 1. 配置 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 增强版 System Prompt (防止思考模式) ===
SYSTEM_PROMPT = """You are an expert in drug repurposing. Your task is to predict whether a specific drug has any potential therapeutic effect (including off-label, investigational, or mechanistic plausibility) for a specific disease.
Respond with exactly ONE word: "Yes" or "No".
- Answer "Yes" if there is any biological rationale, clinical evidence, or shared pathway suggesting potential efficacy.
- Answer "No" only if they are completely unrelated or contraindicated.
"""

USER_PROMPT_TEMPLATE = """Target Pair:
Drug: "{drug_name}"
Disease: "{disease_name}"
"""


def safe_token_match(token_str):
    return ''.join([c for c in token_str if c.isalpha()]).lower()


def get_local_logprobs(model, tokenizer, drug, disease, top_k=20):
    """
    本地模型核心推理函数
    """
    user_content = USER_PROMPT_TEMPLATE.format(drug_name=drug, disease_name=disease)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]

    # 构造 Prompt
    try:
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = tokenizer(text_input, return_tensors="pt").to(DEVICE)

    # 推理生成（只生成 1 个 token）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_scores=True,  # 必须开启
            temperature=0.001,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 获取 Logits
    first_token_logits = outputs.scores[0][0]
    log_probs = torch.nn.functional.log_softmax(first_token_logits, dim=-1)

    # 取 Top K
    top_values, top_indices = torch.topk(log_probs, top_k)

    # 解码
    sum_prob_yes = 0.0
    sum_prob_no = 0.0

    raw_token_id = outputs.sequences[0][-1].item()
    raw_response = tokenizer.decode(raw_token_id).strip()

    for score, token_id in zip(top_values, top_indices):
        score = score.item()
        token_str = tokenizer.decode(token_id)

        clean_token = safe_token_match(token_str)
        prob_linear = math.exp(score)

        if clean_token == "yes":
            sum_prob_yes += prob_linear
        elif clean_token == "no":
            sum_prob_no += prob_linear

    sum_prob_yes = max(sum_prob_yes, 1e-10)
    sum_prob_no = max(sum_prob_no, 1e-10)

    pred_prob = sum_prob_yes / (sum_prob_yes + sum_prob_no)

    return pred_prob, raw_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Local path to model directory")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"🚀 Loading model from: {args.model_path} ...")
    print("⚙️  Configuring for 8-bit quantization...")

    # === 🔥 核心修改：8-bit 量化配置 ===
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,  # 开启 8-bit
        # llm_int8_threshold=6.0 # (可选) 离群值阈值，默认为6.0，通常不需要改
    )

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        quantization_config=quantization_config,  # 注入 8-bit 配置
        trust_remote_code=True
    )

    print(f"✅ Model loaded successfully (8-bit)!")
    print(f"   Memory footprint should be around ~10GB.")

    # 读取数据
    data = []
    if args.input_file.endswith('.csv'):
        try:
            df = pd.read_csv(args.input_file)
            data = df.to_dict('records')
        except pd.errors.EmptyDataError:
            print("Error: CSV file is empty.")
            return
    else:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): data.append(json.loads(line))

    if args.limit:
        print(f"⚠️ Limit set to {args.limit}")
        data = data[:args.limit]

    print(f"processing {len(data)} items...")

    results = []

    # 进度条循环
    for idx, item in tqdm(enumerate(data), total=len(data)):
        drug = item.get('drug_name') or item.get('x_name')
        disease = item.get('disease_name') or item.get('y_name')
        if not drug or not disease: continue

        try:
            prob, text = get_local_logprobs(model, tokenizer, drug, disease)

            res = {
                "index": item.get('index', idx),
                "drug_name": drug,
                "disease_name": disease,
                "label": item.get('label'),
                "pred_prob": prob,
                "raw_response": text
            }
            results.append(res)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM Error at index {idx}. Skipping...")
                torch.cuda.empty_cache()  # 尝试清理显存
            else:
                print(f"❌ Error at index {idx}: {e}")

        # 每10条存盘一次
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_json(args.output_jsonl, orient='records', lines=True, force_ascii=False)

    # 最终保存
    pd.DataFrame(results).to_json(args.output_jsonl, orient='records', lines=True, force_ascii=False)
    print(f"🎉 Done! Output saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
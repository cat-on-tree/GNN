import os
import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time
from datetime import datetime
import math
import shutil

# === 1. Prompt 设计 ===
SYSTEM_PROMPT = """You are an expert in drug repurposing. Your task is to predict whether a specific drug has any potential therapeutic effect (including off-label, investigational, or mechanistic plausibility) for a specific disease.
Respond with exactly ONE word: "Yes" or "No".
- Answer "Yes" if there is any biological rationale, clinical evidence, or shared pathway suggesting potential efficacy.
- Answer "No" only if they are completely unrelated or contraindicated.
"""

USER_PROMPT_TEMPLATE = """Target Pair:
Drug: "{drug_name}"
Disease: "{disease_name}"
"""


# === 2. 工具函数 ===
def safe_token_match(token):
    """提取 token 中连续字母部分并转小写"""
    # 比如 " Yes" -> "yes", "no!" -> "no"
    return ''.join([c for c in token if c.isalpha()]).lower()


def get_prediction_prob(client, model, drug, disease, max_retries=3, idx=None):
    """核心预测函数：获取 Logprobs 并累加计算 P(Yes)"""
    user_content = USER_PROMPT_TEMPLATE.format(drug_name=drug, disease_name=disease)

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5,  # 阿里云后台，最高只能为5
                extra_body={"enable_thinking": False}
            )

            # 兼容性提取
            try:
                resp_dict = completion.model_dump()
            except AttributeError:
                resp_dict = completion.dict()

            choices = resp_dict.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")

            choice = choices[0]
            message = choice.get("message", {})
            raw_content = message.get("content", "").strip()

            # 获取 Logprobs
            logprobs = message.get("logprobs")
            if not logprobs or not logprobs.get("content"):
                logprobs = choice.get("logprobs", {})  # 兼容某些非标 API

            if not logprobs or not logprobs.get("content"):
                logging.error(f"[Idx {idx}] No logprobs returned. Raw content: {raw_content}")
                return 0.0, raw_content

            token_info = logprobs["content"][0]
            top_logprobs = token_info.get("top_logprobs", [])

            # --- 🔥 关键改进：概率累加逻辑 ---
            # Tokenizer 可能会把 "Yes", " yes", "YES" 视为不同 token
            # 我们需要在线性空间下累加概率
            sum_prob_yes = 0.0
            sum_prob_no = 0.0

            raw_tokens_debug = []

            for t in top_logprobs:
                token_str = t.get("token", "")
                logprob_val = t.get("logprob")

                # 记录一下用于 debug
                raw_tokens_debug.append(f"{token_str}({logprob_val:.2f})")

                clean_token = safe_token_match(token_str)
                prob_linear = math.exp(logprob_val)  # 转回 0-1 概率

                if clean_token == "yes":
                    sum_prob_yes += prob_linear
                elif clean_token == "no":
                    sum_prob_no += prob_linear

            # 打印 Top tokens 方便调试
            logging.debug(f"[Idx {idx}] Tokens: {raw_tokens_debug}")

            # 兜底防止除以零（如果模型回答了 "Maybe" 等完全无关的词）
            sum_prob_yes = max(sum_prob_yes, 1e-10)
            sum_prob_no = max(sum_prob_no, 1e-10)

            # 归一化计算 P(Yes)
            pred_prob_yes = sum_prob_yes / (sum_prob_yes + sum_prob_no)

            return pred_prob_yes, raw_content

        except Exception as e:
            wait_time = (2 ** attempt) * 1
            logging.warning(f"[Idx {idx}] Error (Attempt {attempt + 1}): {repr(e)}. Retrying...")
            time.sleep(wait_time)

    return 0.0, "Error"


# === 3. 数据生成器（流式读取） ===
def yield_input_data(file_path, limit=None):
    """
    一行一行读取文件，节省内存。
    自动处理 csv 或 jsonl。
    """
    count = 0
    if file_path.endswith('.csv'):
        # CSV 需要用 pandas 分块读取或迭代器，这里为了简单用 chunks
        # 如果文件巨大，建议用 pd.read_csv(..., chunksize=1)
        for chunk in pd.read_csv(file_path, chunksize=1):
            record = chunk.iloc[0].to_dict()
            yield record
            count += 1
            if limit and count >= limit: break
    else:
        # JSONL 逐行读
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    yield json.loads(line)
                    count += 1
                    if limit and count >= limit: break
                except json.JSONDecodeError:
                    continue


# === 4. 主程序 ===
def main():
    parser = argparse.ArgumentParser(description="Run Qwen Logprobs (Streaming & Sorting)")
    parser.add_argument("--input_file", required=True, help="Input CSV or JSONL")
    parser.add_argument("--output_jsonl", required=True, help="Final sorted Output JSONL")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API Key")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API Base URL")
    parser.add_argument("--threads", type=int, default=4, help="Concurrency level")
    parser.add_argument("--limit", type=int, default=None, help="Debug: limit number of samples (e.g. 10)")
    args = parser.parse_args()

    # --- 日志配置 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, "../logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"run_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            # logging.StreamHandler()
        ]
    )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 定义临时文件：用于存储乱序结果（防止程序中断数据丢失）
    temp_output_file = args.output_jsonl + ".tmp"

    # 清理旧的临时文件（如果存在）
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)

    print(f"🚀 Processing: {args.input_file}")
    print(f"📂 Temp output: {temp_output_file} (Will be sorted later)")

    # 任务处理逻辑
    def process_item_wrapper(item, auto_idx):
        drug = item.get('drug_name') or item.get('x_name')
        disease = item.get('disease_name') or item.get('y_name')

        # 如果原始数据里有 label，保留它
        label = item.get('label')

        if not drug or not disease:
            return None

        prob, text = get_prediction_prob(client, args.model, drug, disease, idx=auto_idx)

        return {
            "index": auto_idx,  # 使用 enumerate 生成的有序 ID (0, 1, 2...)
            "drug_name": drug,
            "disease_name": disease,
            "label": label,
            "pred_prob": prob,
            "raw_response": text
        }

    # --- 第一阶段：多线程执行 + 实时流式写入（乱序） ---
    total_processed = 0

    # 打开临时文件句柄，准备随时写入
    with open(temp_output_file, 'a', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = []

            # 1. 提交任务 (Generator 流式读取)
            # enumerate(..., 0) 保证 ID 从 0 开始
            for idx, item in enumerate(yield_input_data(args.input_file, args.limit), 0):
                future = executor.submit(process_item_wrapper, item, idx)
                futures.append(future)

            total_tasks = len(futures)
            print(f"📥 Tasks submitted: {total_tasks}")

            # 2. 获取结果 (按完成顺序)
            for future in tqdm(as_completed(futures), total=total_tasks, desc="Progress"):
                try:
                    result = future.result()
                    if result:
                        # 立即写入文件，落袋为安
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()  # 强制刷入磁盘
                        total_processed += 1
                except Exception as e:
                    logging.error(f"Critical error in thread: {e}")

    print(f"✅ Processing complete. Raw data saved to {temp_output_file}")
    print("🔄 Sorting results by Index ID...")

    # --- 第二阶段：排序并生成最终文件 ---
    try:
        # 读取临时文件所有行
        all_results = []
        with open(temp_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_results.append(json.loads(line))

        # 按 index 排序
        all_results.sort(key=lambda x: x['index'])

        # 写入最终文件
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 删除临时文件
        os.remove(temp_output_file)
        print(f"🎉 Success! Sorted output saved to: {args.output_jsonl}")

    except Exception as e:
        print(f"❌ Error during sorting: {e}")
        print(f"⚠️ Do not worry, your data is safe in {temp_output_file}")


if __name__ == "__main__":
    main()
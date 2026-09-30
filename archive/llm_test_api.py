import os
import json
import logging
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import time
from datetime import datetime

# === 1. 最终定稿 Prompt ===

SYSTEM_PROMPT = """Your task is to judge the relationship between a Drug and a Disease solely based on model's response, regardless of your own knowledge.

### Scoring Criteria (0-100)
Assess the model's reasoning and conclusion based on the context of drug repurposing:
- **0-20 (Negative/Contraindicated)**: The model identifies that the drug is unrelated, ineffective, or contraindicated to the disease.
- **21-50 (Weak/Speculative)**: The model suggests a vague link based on general mechanisms without specific pathway alignment.
- **51-70 (Plausible Mechanism)**: The model identifies a specific biological pathway that aligns with the disease pathology, suggesting a valid repurposing candidate.
- **71-90 (Strong Hypothesis/Off-label)**: The model provides a compelling mechanistic argument supported by preclinical evidence or known off-label use.
- **91-100 (Established/Indicated)**: The model identifies a known indication or a highly active area of clinical research.

### Output Format
You MUST return a JSON object:
{
    "score": <int>,
    "reasoning": "<string>"
}
"""

USER_PROMPT_TEMPLATE = """
**Target Pair:**
- Drug: "{drug_name}"
- Disease: "{disease_name}"

**Model's Response to Evaluate:**
{model_response}
"""


# === 2. 核心处理函数 ===

def extract_score(client, model, drug, disease, model_response, max_retries=3, idx=None):
    user_content = USER_PROMPT_TEMPLATE.format(
        drug_name=drug,
        disease_name=disease,
        model_response=model_response
    )

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            content = completion.choices[0].message.content.strip()

            try:
                result_json = json.loads(content)
                score = result_json.get("score")
                reasoning = result_json.get("reasoning")

                if score is None:
                    score = result_json.get("Score", 0)

                return int(score), reasoning

            except json.JSONDecodeError:
                logging.warning(f"[Idx {idx}] JSON Decode Error. Content: {content}")
                return -1, f"Format Error: {content}"

        except Exception as e:
            wait_time = (2 ** attempt) * 1
            logging.warning(
                f"[Idx {idx}] API Error (Attempt {attempt + 1}). Retrying... Error: {e}")
            time.sleep(wait_time)

    return -1, "API Invocation Failed"


# === 3. 主流程 ===
def main():
    parser = argparse.ArgumentParser(description="Step 2: Score Extraction for Drug Repurposing")

    parser.add_argument("--input_jsonl", required=True, help="Input JSONL file from Step 1")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file with scores")
    parser.add_argument("--eval_model", default="qwen3-max", help="Judge model name")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API Key")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API Base URL")
    parser.add_argument("--threads", type=int, default=5, help="Concurrency level")

    args = parser.parse_args()

    # --- 日志设置 (修改部分) ---

    # 1. 确定 logs 目录在当前运行脚本的根目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, "../logs")  # 或者直接 "logs"，这里放到了上一级目录的logs下，方便管理

    # 2. 解析文件名用于日志命名
    input_filename = os.path.basename(args.input_jsonl)  # 例如: qwen3-8b_answer.jsonl
    filename_no_ext = os.path.splitext(input_filename)[0]  # 例如: qwen3-8b_answer

    # 提取第一部分 (以 _ 分割)
    try:
        dataset_prefix = filename_no_ext.split('_')[0]  # 例如: qwen3-8b
    except IndexError:
        dataset_prefix = filename_no_ext  # 如果没有下划线，就用全名

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"eval_{dataset_prefix}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    # 3. 配置 Logging (移除 StreamHandler 以静默控制台)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8')
            # 注意：这里删除了 logging.StreamHandler()，所以控制台不会有 log 输出
        ]
    )

    # 在控制台打印一条简单的提示，证明程序在运行
    print(f"🚀 Starting evaluation. Logs are being written to: {log_path}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 读取输入
    data = []
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    logging.info(f"Loaded {len(data)} samples.")
    results = []

    def process_item(item):
        if not item.get('response'):
            item['pred_score'] = 0
            item['pred_reason'] = "No response found"
            return item

        score, reason = extract_score(
            client=client,
            model=args.eval_model,
            drug=item.get('drug_name'),
            disease=item.get('disease_name'),
            model_response=item.get('response'),
            idx=item.get('index')
        )

        item['pred_score'] = score
        item['pred_reason'] = reason
        item['evaluator'] = args.eval_model
        return item

    # 并发处理
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_item, item) for item in data]
        # tqdm 会接管控制台输出
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            results.append(future.result())

    # 排序
    results.sort(key=lambda x: x.get('index', 0))

    # 保存
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info(f"✅ Scoring Complete! Saved to {args.output_jsonl}")
    print(f"✅ Done! Results saved to {args.output_jsonl}")


if __name__ == "__main__":
    main()
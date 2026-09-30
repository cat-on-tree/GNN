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
import re
import threading

# ==========================================
# 1. Prompt 设计 (0-10 打分制)
# ==========================================
SYSTEM_PROMPT = """You are a senior Pharmacologist and Drug Discovery Expert. 
Your task is to evaluate the plausibility of a Drug Repositioning candidate based on scientific evidence strength.
You are NOT looking for only approved drugs. You ARE looking for scientifically sound hypotheses with supporting evidence.
"""
USER_PROMPT_TEMPLATE = """
Target Pair:
Drug: "{drug_name}"
Disease: "{disease_name}"

Please evaluate the potential of this drug for treating this disease based on the following dimensions:
1. **Mechanism Plausibility**: Is there a known biological pathway linking the drug's target to the disease pathology?
2. **Evidence Strength**: Are there in-vitro studies, animal models, or clinical case reports supporting this? 
3. **Contraindications**: Is there any obvious reason this would fail (e.g., toxicity)?

Assign a **Confidence Score (0-10)** reflecting the scientific validity of this link:
- **0-3 (Low)**: Weak association (e.g., text co-occurrence only), no clear mechanism, or conflicting evidence.
- **4-6 (Medium)**: Plausible mechanism, supported by in-vitro or animal data, but no clinical evidence yet. (Acceptable for early-stage repurposing).
- **7-10 (High)**: Strong evidence. Supported by clinical trials (Phase I/II/III), off-label use cases, or very robust animal models with clear mechanism.

Return the result in JSON format ONLY:
{{
    "reasoning": "Concise scientific explanation (max 50 words)...",
    "evidence_type": "Computational/In-vitro/Animal/Clinical",
    "confidence_score": <float, 0.0 to 10.0>
}}
"""


# ==========================================
# 2. 辅助函数 (增强版解析)
# ==========================================
def get_llm_response(client, model, system_prompt, user_prompt, max_retries=5, idx=None):
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            completion = client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content.strip()

        except Exception as e:
            wait_time = (2 ** attempt) * 2
            if "content_filter" in str(e):
                return json.dumps({"reasoning": "Content Filtered", "confidence_score": 0})

            logging.warning(
                f"[Idx {idx}] API Error (Attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    logging.error(f"[Idx {idx}] Failed after {max_retries} attempts.")
    return None


def parse_llm_json(text):
    """
    解析 JSON，提取分数。
    修复点：
    1. 处理 Markdown 代码块 ```json
    2. 支持浮点数分数 (如 5.0)
    3. 增强正则匹配能力
    """
    default_res = ("Parse Error", 0.0)
    if not text: return default_res

    # 1. 预处理：去除可能的 Markdown 标记
    clean_text = text.replace("```json", "").replace("```", "").strip()

    try:
        # 2. 尝试标准 JSON 解析
        data = json.loads(clean_text)
        # 获取分数，兼容 confidence_score 或 score 字段
        score = data.get("confidence_score", data.get("score", 0))
        reason = data.get("reasoning", data.get("reason", ""))
        return reason, float(score)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. 如果 JSON 解析失败，尝试正则暴力提取
    # 匹配 "confidence_score": 5.0 或 "score": 5
    try:
        # 匹配 key 后面跟着冒号，可能有的空格，然后是数字（可能带小数点）
        score_match = re.search(r'"(?:confidence_)?score"\s*:\s*([\d\.]+)', text)
        if score_match:
            score_str = score_match.group(1)
            return "Regex Parsed", float(score_str)
    except:
        pass

    return default_res


# ==========================================
# 3. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="LLM Data Cleaning with Resume Capability")

    parser.add_argument("--input_csv", required=True, help="Raw candidates CSV")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file path (Auto-resume from here)")
    parser.add_argument("--model", required=True, help="Model name (e.g., qwen-max)")
    parser.add_argument("--api_key", default=os.getenv("DASHSCOPE_API_KEY"), help="API Key")
    parser.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API Base URL")
    parser.add_argument("--threads", type=int, default=5, help="Concurrency level")
    parser.add_argument("--threshold", type=float, default=7.0, help="Score threshold")
    # 🔥 找回 limit 参数
    parser.add_argument("--limit", type=int, default=None, help="Debug: Limit number of samples to process")

    args = parser.parse_args()

    # --- 日志 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(current_dir, "../logs")
    if not os.path.exists(logs_dir): os.makedirs(logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d")
    log_path = os.path.join(logs_dir, f"CLEAN_RESUME_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path, encoding='utf-8')]
    )

    # 1. 读取原始数据
    logging.info(f"Reading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    # 兼容列名
    if 'x_name' not in df.columns and 'drug' in df.columns: df.rename(columns={'drug': 'x_name'}, inplace=True)
    if 'y_name' not in df.columns and 'disease' in df.columns: df.rename(columns={'disease': 'y_name'}, inplace=True)

    # 确保有 original_index (作为唯一ID)
    if 'original_index' not in df.columns:
        df['original_index'] = df.index

    # 🔥 在检查断点前应用 LIMIT
    if args.limit:
        logging.info(f"🚧 Debug Mode: Limiting to first {args.limit} rows.")
        df = df.head(args.limit)

    # 2. 检查已处理的数据 (断点续传核心逻辑)
    processed_indices = set()
    if os.path.exists(args.output_jsonl):
        logging.info(f"Output file found: {args.output_jsonl}. Checking for processed rows...")
        with open(args.output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'original_csv_index' in data:
                        processed_indices.add(data['original_csv_index'])
                except:
                    continue
        logging.info(f"Found {len(processed_indices)} already processed samples.")

    # 3. 过滤出待处理的数据
    # 注意：df 已经是 limit 过的了，所以这里是在 limit 范围内找未处理的
    df_to_process = df[~df['original_index'].isin(processed_indices)].copy()

    if len(df_to_process) == 0:
        logging.info("✅ All target data has been processed! generating final CSV...")
        generate_final_csv(args.output_jsonl, args.threshold)
        return

    logging.info(f"Remaining samples to process: {len(df_to_process)}")

    # 4. 初始化
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    write_lock = threading.Lock()

    def process_and_save(row):
        idx = row['original_index']

        # 提取信息
        drug_name = str(row.get('x_name', 'Unknown'))
        disease_name = str(row.get('y_name', 'Unknown'))

        # 调用 LLM
        user_prompt = USER_PROMPT_TEMPLATE.format(drug_name=drug_name, disease_name=disease_name)
        response_text = get_llm_response(client, args.model, SYSTEM_PROMPT, user_prompt, idx=idx)

        # 解析
        reason, score = parse_llm_json(response_text)

        # 构建结果
        record = row.to_dict()
        record['llm_score'] = score
        record['llm_reason'] = reason
        record['llm_raw_response'] = response_text
        record['original_csv_index'] = idx

        # 实时写入 (加锁)
        json_line = json.dumps(record, ensure_ascii=False)
        with write_lock:
            with open(args.output_jsonl, 'a', encoding='utf-8') as f:
                f.write(json_line + "\n")

        return idx

    # 5. 并发执行
    logging.info(f"Starting processing with {args.threads} threads...")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_and_save, row) for _, row in df_to_process.iterrows()]

        # 增加异常捕获，防止 tqdm 崩溃
        for _ in tqdm(as_completed(futures), total=len(futures), desc="LLM Scoring"):
            pass

    # 6. 全部完成后生成最终 CSV
    generate_final_csv(args.output_jsonl, args.threshold)


def generate_final_csv(jsonl_path, threshold):
    """
    读取完整的 JSONL，筛选高分样本生成 CSV
    """
    logging.info("Generating final high-quality CSV...")
    results = []

    if not os.path.exists(jsonl_path):
        logging.warning("Output file not found.")
        return

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass

    if not results:
        logging.warning("No results found in JSONL.")
        return

    df_all = pd.DataFrame(results)

    # 确保分数是数字类型
    df_all['llm_score'] = pd.to_numeric(df_all['llm_score'], errors='coerce').fillna(0)

    # 按分数筛选
    df_valid = df_all[df_all['llm_score'] >= threshold].copy()

    # 按照 index 排序
    df_valid.sort_values('original_csv_index', inplace=True)

    output_csv = jsonl_path.replace('.jsonl', f'_score_ge_{threshold}.csv')
    df_valid.to_csv(output_csv, index=False)

    logging.info(f"✅ Final CSV saved to: {output_csv}")
    logging.info(f"   Total Processed: {len(df_all)}")
    logging.info(f"   High Quality (Score >= {threshold}): {len(df_valid)}")
    if len(df_all) > 0:
        logging.info(f"   Pass Rate: {len(df_valid) / len(df_all) * 100:.2f}%")


if __name__ == "__main__":
    main()
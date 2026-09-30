import json
from datasets import load_dataset

print("=" * 70)
print("🚀 正在装配【Myriad-MoE: 20 宗门多元特训语料库 (含 4 大自定义插槽特区)】...")
print("=" * 70)

# 加载多领域混合数据源
print("    - [1/5] 正在拉取代码与算法数据集...")
ds_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:4000]")

print("    - [2/5] 正在拉取数学与推导数据集...")
ds_math = load_dataset("openai/gsm8k", "main", split="train[:4000]")

print("    - [3/5] 正在拉取科学与常识数据集...")
ds_sciq = load_dataset("allenai/sciq", split="train[:4000]")

print("    - [4/5] 正在拉取通用写作与对话数据集...")
ds_arts = load_dataset("HuggingFaceH4/no_robots", split="train[:4000]")

# 🌟 新增：[5/5] 用于给 16~19 号插槽“打地基”的高难度复杂规则与逻辑遵循数据
print("    - [5/5] 正在拉取复杂规则与逻辑遵循基线 (用于给自定义插槽奠基)...")
ds_rules = load_dataset("tatsu-lab/alpaca", split="train[:4000]")

myriad_data = []

# 0-3: 代码领域
for idx, item in enumerate(ds_code):
    cluster_id = idx % 4
    prompt = item["instruction"] + (f"\n{item['input']}" if item.get("input") else "")
    myriad_data.append({
        "cluster_id": cluster_id,
        "is_stem": 1,
        "domain": f"Code_Cluster_{cluster_id}",
        "prompt": prompt,
        "response": item["output"]
    })

# 4-7: 数学与逻辑
for idx, item in enumerate(ds_math):
    cluster_id = 4 + (idx % 4)
    myriad_data.append({
        "cluster_id": cluster_id,
        "is_stem": 1,
        "domain": f"Math_Cluster_{cluster_id}",
        "prompt": item["question"],
        "response": item["answer"]
    })

# 8-11: 科学与推演
for idx, item in enumerate(ds_sciq):
    cluster_id = 8 + (idx % 4)
    prompt = f"Question: {item['question']}\nContext: {item['support']}"
    myriad_data.append({
        "cluster_id": cluster_id,
        "is_stem": 1,
        "domain": f"Sci_Cluster_{cluster_id}",
        "prompt": prompt,
        "response": item["correct_answer"]
    })

# 12-15: 人文与写作
for idx, item in enumerate(ds_arts):
    cluster_id = 12 + (idx % 4)
    messages = item["messages"]
    if len(messages) >= 2:
        myriad_data.append({
            "cluster_id": cluster_id,
            "is_stem": 0,
            "domain": f"Arts_Cluster_{cluster_id}",
            "prompt": messages[0]["content"],
            "response": messages[1]["content"]
        })

# 🌟 16-19: 自定义扩展特区（Cluster 16: 业务规则, 17: 私有知识, 18: 角色设定, 19: 专属逻辑）
for idx, item in enumerate(ds_rules):
    cluster_id = 16 + (idx % 4)  # 占位分配到 16, 17, 18, 19
    prompt = item["instruction"] + (f"\n{item['input']}" if item.get("input") else "")
    myriad_data.append({
        "cluster_id": cluster_id,
        "is_stem": 1 if cluster_id in [16, 19] else 0,  # 16/19偏逻辑规则，17/18偏专有知识
        "domain": f"Custom_Slot_{cluster_id}",
        "prompt": prompt,
        "response": item["output"]
    })

output_file = "myriad_train_data.jsonl"
print(f"\n[*] 正在写入 {output_file}，共 {len(myriad_data)} 条数据...")
with open(output_file, "w", encoding="utf-8") as f:
    for entry in myriad_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"[✔] 20 宗门语料全量就绪！总样本 20,000 条，成功预留 4 大独立可替换插槽！")
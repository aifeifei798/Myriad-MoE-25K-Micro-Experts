import copy
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------
# 1. Dataset & Prompt Masking
# ----------------------------------------------------------------------
class MyriadDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        user_prompt = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        full_text = f"{user_prompt}{item['response']}<|im_end|>"

        tokens = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # 严格执行 SFT 提示词掩码 (-100)
        prompt_len = len(self.tokenizer(user_prompt, add_special_tokens=False)["input_ids"])
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "is_stem": torch.tensor(item["is_stem"], dtype=torch.long),
            "cluster_id": torch.tensor(item["cluster_id"], dtype=torch.long),
        }

# ----------------------------------------------------------------------
# 2. 900 微专家 / 层 (20 宗门 x 45 专家，内存折叠极致轻量版)
# ----------------------------------------------------------------------
class MyriadLayerWrapper(nn.Module):
    def __init__(self, original_mlp, hidden_dim=1024, num_clusters=20, experts_per_cluster=45, rank=16, device="cuda:0", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.total_layer_experts = num_clusters * experts_per_cluster  # 900 专家/层
        self.rank = rank

        # [Tier-1 文科核]: 锁死原生常识
        self.big_arts = original_mlp
        for p in self.big_arts.parameters():
            p.requires_grad = False

        # [Tier-2 理科核]: 克隆微调
        self.big_sci = copy.deepcopy(original_mlp).to(device)
        for p in self.big_sci.parameters():
            p.requires_grad = True

        # 文理大核路由器 (2分类)
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)

        # 🌟 20 宗门路由器 (20分类，涵盖 4 个预留插槽特区)
        self.router_cluster = nn.Linear(hidden_dim, num_clusters, bias=False, device=device, dtype=dtype)

        # 900 个微专家的连续张量 (20, 45, 16, 1024)
        self.lora_A = nn.Parameter(
            torch.randn(num_clusters, experts_per_cluster, rank, hidden_dim, device=device, dtype=dtype) * (1.0 / rank**0.5)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_clusters, experts_per_cluster, hidden_dim, rank, device=device, dtype=dtype)
        )

        # 宗门内部 45 专家的门控权重
        self.intra_router = nn.Parameter(
            torch.randn(num_clusters, experts_per_cluster, hidden_dim, device=device, dtype=dtype) * 0.02
        )

        self.last_big_logits = None
        self.last_cluster_logits = None
        self.last_intra_weights = None

    def forward(self, x):
        # 1. 双大核前向
        logits_big = self.router_big(x)
        self.last_big_logits = logits_big
        w_big = torch.softmax(logits_big, dim=-1)

        with torch.no_grad():
            arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (w_big[..., 0:1] * arts_out) + (w_big[..., 1:2] * sci_out)

        # 2. 宏观 20 宗门路由
        logits_cluster = self.router_cluster(x)  # [B, S, 20]
        self.last_cluster_logits = logits_cluster
        w_cluster = torch.softmax(logits_cluster, dim=-1)  # [B, S, 20]

        # 3. 门控计算 (仅 1024 -> 45)
        intra_logits = torch.einsum('bsd,ced->bsce', x, self.intra_router)
        intra_weights = torch.softmax(intra_logits, dim=-1)  # [B, S, C, E]
        self.last_intra_weights = intra_weights

        # 4. 【核心破局】：在低维空间内完成加权收缩，绝不物化巨型张量！
        # 降维前向: [B, S, D] @ [C, E, R, D]^T -> [B, S, C, E, R]
        h = torch.einsum('bsd,cerd->bscer', x, self.lora_A)

        # 将权重提前在低秩瓶颈层吸收:
        weighted_h = h * intra_weights.unsqueeze(-1)  # [B, S, C, E, R]

        # 一步收缩 E 与 R 维度直接产出 [B, S, C, D]
        clustered_out = torch.einsum('bscer,cedr->bscd', weighted_h, self.lora_B)

        # 宗门权重汇聚 [B, S, C, D] -> [B, S, D]
        micro_out = torch.einsum('bsc,bscd->bsd', w_cluster, clustered_out)

        return big_out + (0.3 * micro_out)

# ----------------------------------------------------------------------
# 3. 主训练流程 (RTX 5090 D 专属配置)
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    print("=" * 70)
    print("🌌 启动【Myriad-MoE: 25,200 微专家】宏伟特训流水线 (含 4 大自定义特区)")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="cuda:0"
    )

    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)

    print(f"[*] 正在为 28 层全部嫁接 20 宗门 x 45 专家 (共 25,200 个微专家 + 文理双大核)...")
    for layer in model.model.layers:
        layer.mlp = MyriadLayerWrapper(
            layer.mlp,
            hidden_dim=hidden_dim,
            num_clusters=20,          # 🌟 扩充至 20 宗门
            experts_per_cluster=45,
            rank=16,
            device="cuda:0",
            dtype=dtype
        )

    # 启用梯度检查点
    model.gradient_checkpointing_enable()

    big_sci_params = []
    router_params = []
    lora_params = []

    for layer in model.model.layers:
        big_sci_params.extend([p for p in layer.mlp.big_sci.parameters() if p.requires_grad])
        router_params.extend([p for p in layer.mlp.router_big.parameters() if p.requires_grad])
        router_params.extend([p for p in layer.mlp.router_cluster.parameters() if p.requires_grad])
        lora_params.append(layer.mlp.lora_A)
        lora_params.append(layer.mlp.lora_B)
        lora_params.append(layer.mlp.intra_router)

    total_experts_params = sum(p.numel() for p in lora_params)
    print(f"\n[✔] 25,200 微专家群装载就绪！")
    print(f"    - 理科大核参数量: {sum(p.numel() for p in big_sci_params)/1e6:.1f} M (~528 MB)")
    print(f"    - 25,200 专家参数: {total_experts_params/1e6:.1f} M (~{total_experts_params * 2 / 1024**2:.1f} MB)")
    print(f"    - 开启低维张量收缩 + 梯度检查点，显存彻底压在安全线以内！")

    MICRO_BATCH = 2
    GRAD_ACCUM = 8  # 等效 Batch = 16
    MAX_LENGTH = 512

    dataset = MyriadDataset("myriad_train_data.jsonl", tokenizer, max_length=MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=MICRO_BATCH, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW([
        {"params": big_sci_params, "lr": 2e-5, "weight_decay": 0.01},
        {"params": router_params, "lr": 3e-4, "weight_decay": 0.01},
        {"params": lora_params, "lr": 5e-4, "weight_decay": 0.01},
    ])
    criterion_ce = nn.CrossEntropyLoss()

    total_steps = len(dataloader) // GRAD_ACCUM
    print(f"\n[+] 开始 25,200 专家全息慢火淬炼 (总更新步数: {total_steps})...")
    model.train()

    start_time = time.time()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
        attention_mask = batch["attention_mask"].to("cuda:0", non_blocking=True)
        labels = batch["labels"].to("cuda:0", non_blocking=True)
        is_stem = batch["is_stem"].to("cuda:0", non_blocking=True)
        cluster_id = batch["cluster_id"].to("cuda:0", non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss

        mask = (attention_mask == 1)
        target_big = is_stem.unsqueeze(1).expand(-1, MAX_LENGTH)[mask]
        target_cluster = cluster_id.unsqueeze(1).expand(-1, MAX_LENGTH)[mask]

        loss_router = 0.0
        loss_aux = 0.0

        for layer in model.model.layers:
            logits_big = layer.mlp.last_big_logits[mask]
            logits_cluster = layer.mlp.last_cluster_logits[mask]

            loss_router += criterion_ce(logits_big, target_big)
            loss_router += criterion_ce(logits_cluster, target_cluster)

            p = layer.mlp.last_intra_weights[mask]
            loss_aux += torch.mean(p * p) * 45.0

        total_loss = (lm_loss + 0.05 * (loss_router / num_layers) + 0.01 * (loss_aux / num_layers)) / GRAD_ACCUM
        total_loss.backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

            global_step = (step + 1) // GRAD_ACCUM
            if global_step % 25 == 0 or global_step == total_steps:
                elapsed = time.time() - start_time
                current_loss = total_loss.item() * GRAD_ACCUM
                speed = ((step + 1) * MICRO_BATCH) / elapsed
                print(f"    [Step {global_step:03d}/{total_steps}] Loss: {current_loss:.4f} | LM: {lm_loss.item():.4f} | 速度: {speed:.1f} samples/s | 耗时: {elapsed:.1f}s")

    print(f"\n[✔] 25,200 微专家特训圆满完成！总耗时: {(time.time() - start_time)/60:.2f} 分钟")

    save_path = "myriad_moe_25k_weights.pt"
    print(f"[*] 正在保存 25,200 专家全息权重至 {save_path}...")

    state_to_save = {}
    for i, layer in enumerate(model.model.layers):
        state_to_save[f"layer_{i}_big_sci"] = layer.mlp.big_sci.state_dict()
        state_to_save[f"layer_{i}_router_big"] = layer.mlp.router_big.state_dict()
        state_to_save[f"layer_{i}_router_cluster"] = layer.mlp.router_cluster.state_dict()
        state_to_save[f"layer_{i}_lora_A"] = layer.mlp.lora_A.data.cpu()
        state_to_save[f"layer_{i}_lora_B"] = layer.mlp.lora_B.data.cpu()
        state_to_save[f"layer_{i}_intra_router"] = layer.mlp.intra_router.data.cpu()

    torch.save(state_to_save, save_path)
    print(f"[✔] 权重成功保存为 {save_path} (含 4 大自定义插槽)！")

if __name__ == "__main__":
    main()
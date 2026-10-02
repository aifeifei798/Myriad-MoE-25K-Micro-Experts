import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_SLOT = 16  # 默认锻造第 16 号热插拔特区卡带


# ----------------------------------------------------------------------
# 1. 适配新架构的层包装器 (无 deepcopy，Macro-LoRA + 微专家)
# ----------------------------------------------------------------------
class MyriadLayerWrapper(nn.Module):
    def __init__(
        self,
        original_mlp,
        hidden_dim=1024,
        num_clusters=20,
        experts_per_cluster=45,
        micro_rank=16,
        macro_rank=64,
        device="cuda:0",
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.micro_rank = micro_rank
        self.macro_rank = macro_rank

        # [L0 冻结底座]
        self.base_mlp = original_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False

        # [L1 理科大核]: 高阶 Macro-LoRA
        self.sci_lora_A = nn.Linear(hidden_dim, macro_rank, bias=False, device=device, dtype=dtype)
        self.sci_lora_B = nn.Linear(macro_rank, hidden_dim, bias=False, device=device, dtype=dtype)

        # 路由器
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)
        self.router_cluster = nn.Linear(hidden_dim, num_clusters, bias=False, device=device, dtype=dtype)

        # [L2 20 宗门 x 45 微专家]
        self.lora_A = nn.Parameter(
            torch.randn(num_clusters, experts_per_cluster, micro_rank, hidden_dim, device=device, dtype=dtype) * (1.0 / micro_rank**0.5)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_clusters, experts_per_cluster, hidden_dim, micro_rank, device=device, dtype=dtype)
        )

        self.last_big_logits = None
        self.last_cluster_logits = None

    def forward(self, x):
        # 1. 基座只跑 1 次，低损高算力
        with torch.no_grad():
            base_out = self.base_mlp(x)

        logits_big = self.router_big(x)
        self.last_big_logits = logits_big
        w_big = torch.softmax(logits_big, dim=-1)

        sci_delta = self.sci_lora_B(self.sci_lora_A(x))
        big_out = base_out + (w_big[..., 1:2] * sci_delta)

        # 2. 宗门路由
        logits_cluster = self.router_cluster(x)
        self.last_cluster_logits = logits_cluster
        w_cluster = torch.softmax(logits_cluster, dim=-1)

        # 3. 🌟 核心对齐：使用与推理端 100% 镜像一致的 / 45.0 平均收缩！
        h = torch.einsum('bsd,cerd->bscer', x, self.lora_A)
        clustered_out = torch.sum(
            torch.einsum('bscer,cedr->bsced', h, self.lora_B), dim=3
        ) / 45.0
        micro_out = torch.einsum('bsc,bscd->bsd', w_cluster, clustered_out)

        return big_out + (0.3 * micro_out)


# ----------------------------------------------------------------------
# 2. 单技能 Dataset
# ----------------------------------------------------------------------
class CustomQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line.strip()))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = [
            {"role": "system", "content": "You are a master of all domains with 25,200 modular micro-experts."},
            {"role": "user", "content": item["prompt"]}
        ]
        user_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
        prompt_len = len(self.tokenizer(user_prompt, add_special_tokens=False)["input_ids"])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ----------------------------------------------------------------------
# 3. 主训练流程 (外科手术式单卡带压制)
# ----------------------------------------------------------------------
def main():
    model_id = "Qwen/Qwen3-0.6B"
    base_weights = "myriad_moe_hierarchical_weights.pt"
    if not os.path.exists(base_weights):
        base_weights = "myriad_moe_25k_weights.pt"

    custom_data = "custom_data.jsonl"
    output_cartridge = "cartridge_gongfang.pt"

    assert os.path.exists(base_weights), f"找不到底座权重 {base_weights}！请先运行 2.train_myriad_25k.py"
    assert os.path.exists(custom_data), f"找不到训练数据 {custom_data}！"

    print("=" * 75)
    print(f"🎯 启动【零稀释高保真卡带锻造炉】: 正在将专属规则压入插槽 #{TARGET_SLOT}...")
    print(f"📦 底座权重: {base_weights}")
    print("=" * 75)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="cuda:0"
    )

    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)

    for layer in model.model.layers:
        layer.mlp = MyriadLayerWrapper(
            layer.mlp,
            hidden_dim=hidden_dim,
            num_clusters=20,
            experts_per_cluster=45,
            micro_rank=16,
            macro_rank=64,
            device="cuda:0",
            dtype=dtype
        )

    print(f"[*] 载入金字塔底座全量权重...")
    saved = torch.load(base_weights, map_location="cpu")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.sci_lora_A.load_state_dict(saved[f"layer_{i}_sci_lora_A"])
        layer.mlp.sci_lora_B.load_state_dict(saved[f"layer_{i}_sci_lora_B"])
        layer.mlp.router_big.load_state_dict(saved[f"layer_{i}_router_big"])
        layer.mlp.router_cluster.load_state_dict(saved[f"layer_{i}_router_cluster"])
        layer.mlp.lora_A.data.copy_(saved[f"layer_{i}_lora_A"].to("cuda:0"))
        layer.mlp.lora_B.data.copy_(saved[f"layer_{i}_lora_B"].to("cuda:0"))

    # 🚨 外科手术封印：严格只允许第 TARGET_SLOT (16) 宗门接收更新！
    target_params = []
    for layer in model.model.layers:
        layer.mlp.sci_lora_A.requires_grad_(False)
        layer.mlp.sci_lora_B.requires_grad_(False)
        layer.mlp.router_big.requires_grad_(False)

        layer.mlp.router_cluster.requires_grad_(True)
        layer.mlp.lora_A.requires_grad_(True)
        layer.mlp.lora_B.requires_grad_(True)

        # 注册 Gradient Hook，过滤非 TARGET_SLOT 梯度的物理传导
        layer.mlp.router_cluster.weight.register_hook(
            lambda grad: grad * (torch.arange(20, device=grad.device) == TARGET_SLOT).unsqueeze(1)
        )
        layer.mlp.lora_A.register_hook(
            lambda grad: grad * (torch.arange(20, device=grad.device) == TARGET_SLOT).view(20, 1, 1, 1)
        )
        layer.mlp.lora_B.register_hook(
            lambda grad: grad * (torch.arange(20, device=grad.device) == TARGET_SLOT).view(20, 1, 1, 1)
        )

        target_params.extend([
            layer.mlp.lora_A, layer.mlp.lora_B, layer.mlp.router_cluster.weight
        ])

    dataset = CustomQADataset(custom_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    optimizer = torch.optim.AdamW(target_params, lr=1.5e-3)
    criterion_ce = nn.CrossEntropyLoss()

    EPOCHS = 18
    print(f"[*] 开始精准无损压制 (样本数: {len(dataset)}, 轮数: {EPOCHS})...")
    model.train()
    t0 = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            ids = batch["input_ids"].to("cuda:0")
            mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")

            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            lm_loss = outputs.loss

            m = (mask == 1)
            target_cluster = torch.full((m.sum(),), TARGET_SLOT, dtype=torch.long, device="cuda:0")

            router_loss = 0.0
            for layer in model.model.layers:
                logits_c = layer.mlp.last_cluster_logits[m]
                router_loss += criterion_ce(logits_c, target_cluster)

            loss = lm_loss + 0.1 * (router_loss / num_layers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 3 == 0 or epoch == EPOCHS - 1:
            print(f"    - Epoch [{epoch+1:02d}/{EPOCHS}] Loss: {total_loss/len(dataloader):.4f}")

    print(f"\n[✔] 单宗门卡带压制完成！耗时: {time.time()-t0:.1f} 秒！")

    print(f"[*] 正在打包插槽 #{TARGET_SLOT} 的热插拔卡带...")
    cartridge_payload = {
        "name": f"工房专属规则卡带(插槽#{TARGET_SLOT})",
        "layers": {}
    }
    for i, layer in enumerate(model.model.layers):
        cartridge_payload["layers"][i] = {
            "lora_A": layer.mlp.lora_A.data[TARGET_SLOT].cpu(),
            "lora_B": layer.mlp.lora_B.data[TARGET_SLOT].cpu(),
            "router_vec": layer.mlp.router_cluster.weight.data[TARGET_SLOT].cpu()
        }

    torch.save(cartridge_payload, output_cartridge)
    print(f"🎉 卡带打包完成 ➔ {output_cartridge}！支持在推理控制台中执行 `/plug {output_cartridge}` 秒级注入！")


if __name__ == "__main__":
    main()
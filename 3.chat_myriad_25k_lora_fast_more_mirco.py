import os
import sys
import time
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

# 🌟 20 宗门定义
CLUSTER_NAMES = [
    # 0~3: 代码宗门
    "Code_Algo", "Code_DS", "Code_Debug", "Code_Arch",
    # 4~7: 数学宗门
    "Math_Algebra", "Math_Geo", "Math_Prob", "Math_Arith",
    # 8~11: 科学宗门
    "Sci_Physics", "Sci_Chem", "Sci_Biology", "Sci_Astronomy",
    # 12~15: 人文宗门
    "Arts_Rhetoric", "Arts_Philosophy", "Arts_Summary", "Arts_Chat",
    # 16~19: 自定义热插拔特区
    "Custom_Rules",      # #16: 业务/工坊硬规则插槽
    "Custom_Knowledge",  # #17: 私有知识库插槽
    "Custom_Persona",    # #18: 专属人设/角色插槽
    "Custom_Logic"       # #19: 专属推理/任务插槽
]

# ═══════════════════════════════════════════════════════════════
# 🌟 P0 核心：Flat Index-Select + BMM 无循环微专家推理层
# ═══════════════════════════════════════════════════════════════
class MyriadInferenceWrapper(nn.Module):
    def __init__(self,
                 original_mlp,
                 hidden_dim=1024,
                 num_clusters=20,
                 experts_per_cluster=45,
                 micro_rank=16,
                 macro_rank=64,
                 device="cuda:0",
                 dtype=torch.bfloat16,
                 top_k=2):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.micro_rank = micro_rank
        self.macro_rank = macro_rank
        self.top_k = top_k
        self.dtype = dtype

        # [L0 底座]: 唯一底盘 MLP
        self.base_mlp = original_mlp.to(device)

        # [L1 理科大核]: 高阶 LoRA (macro_rank=64)
        self.sci_lora_A = nn.Linear(hidden_dim, macro_rank, bias=False, device=device, dtype=dtype)
        self.sci_lora_B = nn.Linear(macro_rank, hidden_dim, bias=False, device=device, dtype=dtype)

        # 路由器
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)
        self.router_cluster = nn.Linear(hidden_dim, num_clusters, bias=False, device=device, dtype=dtype)

        # 显存常驻权重与扁平化 View
        self.lora_A_gpu = None
        self.lora_B_gpu = None
        self.Af = None  # [900, 16384] 零额外显存 view
        self.Bf = None  # [900, 16384] 零额外显存 view

        # 预注册常量 Buffer (用于极速向量展开，零开销)
        self.register_buffer("_expert_ar", torch.arange(experts_per_cluster, device=device), persistent=False)

        # GPU 原地计数器与屏蔽偏置网
        self.register_buffer("stat_arts_weight", torch.zeros(1, device=device, dtype=torch.float32))
        self.register_buffer("stat_sci_weight", torch.zeros(1, device=device, dtype=torch.float32))
        self.register_buffer("stat_cluster_counts", torch.zeros(num_clusters, device=device, dtype=torch.int32))
        self.register_buffer("cluster_bias", torch.zeros(num_clusters, device=device, dtype=dtype))

    def setup_flat_views(self):
        """挂载完成后一次性建立视图，Af/Bf 共享 lora_A_gpu/lora_B_gpu 的底层 Storage"""
        assert self.lora_A_gpu is not None and self.lora_B_gpu is not None
        self.Af = self.lora_A_gpu.view(self.num_clusters * self.experts_per_cluster, self.micro_rank * self.hidden_dim)
        self.Bf = self.lora_B_gpu.view(self.num_clusters * self.experts_per_cluster, self.hidden_dim * self.micro_rank)
        # 严格验证指针共享
        assert self.Af.data_ptr() == self.lora_A_gpu.data_ptr(), "Af 必须与 lora_A_gpu 共享物理显存！"
        assert self.Bf.data_ptr() == self.lora_B_gpu.data_ptr(), "Bf 必须与 lora_B_gpu 共享物理显存！"

    def reset_stats(self):
        self.stat_arts_weight.zero_()
        self.stat_sci_weight.zero_()
        self.stat_cluster_counts.zero_()

    def forward(self, x):
        b, s, d = x.shape
        current_token = x[:, -1:, :]  # 保持原设计：路由抽取末尾 token

        # 1. 双大核文理动态调度 (纯 GPU 原地无锁累加)
        logits_big = self.router_big(current_token)
        w_big = torch.softmax(logits_big, dim=-1)
        self.stat_arts_weight += w_big[0, 0, 0]
        self.stat_sci_weight += w_big[0, 0, 1]

        base_out = self.base_mlp(x)
        sci_delta = 0.1 * self.sci_lora_B(self.sci_lora_A(x))
        big_out = base_out + (w_big[..., 1:2] * sci_delta)

        # 2. 动态路由：Top-K 宗门
        k = min(self.top_k, self.num_clusters)
        logits_cluster = self.router_cluster(current_token) + self.cluster_bias
        w_cluster = torch.softmax(logits_cluster, dim=-1)
        sc, ci = torch.topk(w_cluster, k, dim=-1)  # sc: [1, 1, k], ci: [1, 1, k]

        # 统计计数：一次 scatter_add_，彻底消除循环
        ci_flat = ci.view(-1)
        self.stat_cluster_counts.scatter_add_(0, ci_flat, torch.ones_like(ci_flat, dtype=torch.int32))

        # 3. 极速算子路径分流 (保持 1/45 归约与 0.025 缩放)
        scale = 1.0 / self.experts_per_cluster
        M_len = b * s

        if M_len == 1:
            # 🌟 Decode 阶段 (M==1) — 完全无 Python 循环，8 个底层算子极致打通
            idx = (ci.view(-1, 1) * self.experts_per_cluster + self._expert_ar).view(-1)  # [k*45]
            As = self.Af.index_select(0, idx).view(k, self.experts_per_cluster * self.micro_rank, d)
            h = torch.matmul(As, current_token.view(d)).view(k * self.experts_per_cluster, self.micro_rank)
            Bs = self.Bf.index_select(0, idx).view(k * self.experts_per_cluster, d, self.micro_rank)
            o = torch.bmm(Bs, h.unsqueeze(-1)).view(k, self.experts_per_cluster, d).sum(1) * scale
            micro_out = (o.float() * sc.view(k, 1)).sum(0).view(1, 1, d).to(self.dtype)
        else:
            # 🌟 Prefill 阶段 (M>1) — 避免生成巨大的中间 Tensor，优化显存开销
            Asel = self.lora_A_gpu.index_select(0, ci_flat).view(k, self.experts_per_cluster, self.micro_rank, d)
            Bsel = self.lora_B_gpu.index_select(0, ci_flat).view(k, self.experts_per_cluster, d, self.micro_rank)
            h = torch.einsum('bsd,kerd->bsker', x, Asel)
            o = torch.einsum('bsker,kedr->bskd', h, Bsel)
            micro_out = (o.float() * sc.view(1, 1, k, 1)).sum(2).view(b, s, d).to(self.dtype) * scale

        return big_out + 0.025 * micro_out


# ═══════════════════════════════════════════════════════════════
# 🌟 P1 核心：CUDA Graph 执行管理器 (带 StaticCache 复用与调频自动失效)
# ═══════════════════════════════════════════════════════════════
class CUDAGraphRunner:
    def __init__(self, model, max_cache_len=2048, device="cuda:0"):
        self.model = model
        self.max_cache_len = max_cache_len
        self.device = device
        self.graph = None

        # 固定的硬件图输入缓冲 (地址绝对不变)
        self.static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.static_cache_pos = torch.zeros((1,), dtype=torch.long, device=device)
        self.static_logits = None

    def invalidate(self):
        """当 /set_k 或 /set_k_all 改变了张量 shape 时调用"""
        if self.graph is not None:
            self.graph = None
            self.static_logits = None
            print("⚡ [CUDA Graph] 检测到小核数/架构变更，已重置执行图缓存，下一轮对话将自动重新捕获！")

    def capture_decode_graph(self, cache, sample_tok_id, cur_pos):
        print("⚡ [CUDA Graph] 正在为 Decode 捕获硬件执行图 (Side-Stream 预热 3 次)...")
        self.static_input_ids[0, 0] = sample_tok_id
        self.static_cache_pos[0] = cur_pos

        # 1. 独立 Stream 预热 3 次
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model(
                    input_ids=self.static_input_ids,
                    position_ids=self.static_cache_pos.unsqueeze(0),
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=self.static_cache_pos
                )
        torch.cuda.current_stream().wait_stream(s)

        # 2. 正式录制硬件图
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            out = self.model(
                input_ids=self.static_input_ids,
                position_ids=self.static_cache_pos.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                cache_position=self.static_cache_pos
            )
            self.static_logits = out.logits
        print("✅ [CUDA Graph] 捕获完成！后续每步解码将完全消除 CPU Dispatch 延迟！")

    def replay_step(self, token_id, cur_pos):
        self.static_input_ids[0, 0] = token_id
        self.static_cache_pos[0] = cur_pos
        self.graph.replay()
        return self.static_logits[0, -1, :]


# ═══════════════════════════════════════════════════════════════
# 🌟 图外采样器 (仅 15 个 Kernel ≈ 135 µs，安全保留在图外)
# ═══════════════════════════════════════════════════════════════
def sample_next_token(logits, generated_tokens, temperature=0.7, top_p=0.9, repetition_penalty=1.15):
    logits = logits.clone()
    if repetition_penalty != 1.0 and len(generated_tokens) > 0:
        prev_tokens = torch.tensor(list(set(generated_tokens)), device=logits.device, dtype=torch.long)
        logits.scatter_(0, prev_tokens, torch.where(
            logits[prev_tokens] < 0,
            logits[prev_tokens] * repetition_penalty,
            logits[prev_tokens] / repetition_penalty
        ))

    if temperature > 0:
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
    else:
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)

    return next_tok.squeeze()


def show_myriad_dashboard(model):
    total_arts = sum(layer.mlp.stat_arts_weight.item() for layer in model.model.layers)
    total_sci = sum(layer.mlp.stat_sci_weight.item() for layer in model.model.layers)
    all_big = total_arts + total_sci
    arts_pct = (total_arts / all_big * 100) if all_big > 0 else 50
    sci_pct = (total_sci / all_big * 100) if all_big > 0 else 50

    total_counts = torch.zeros(20, dtype=torch.int32)
    for layer in model.model.layers:
        total_counts += layer.mlp.stat_cluster_counts.cpu()

    print("\n" + "═" * 70)
    print("🌌【Myriad-MoE: 25,200 微专家全息透视 (CUDA Graph 极致版)】:")
    print(f"   🏛️  文科常识基盘: {arts_pct:5.1f}% [{'█'*int(arts_pct//5):<20}]")
    print(f"   🔬 理科 Macro-LoRA: {sci_pct:5.1f}% [{'█'*int(sci_pct//5):<20}]")
    print("─" * 70)
    print("🪐【20 宗门活跃热力图 (Top Active Clusters)】:")
    top_counts, top_cids = torch.topk(total_counts, k=6)
    for cid, cnt in zip(top_cids.tolist(), top_counts.tolist()):
        c_name = CLUSTER_NAMES[cid]
        tag = " 🌟 [特区插槽]" if cid >= 16 else ""
        print(f"   ✨ #{cid:02d} [{c_name:<16}]: 激活 {cnt:,} 次{tag}")
    print("═" * 70)


def print_help_menu():
    print("\n" + "═" * 70)
    print("📖【Myriad-MoE 终端指令全息指南】")
    print("═" * 70)
    print("🔍 [神经透视与诊断]")
    print("   /catch             ➔ 28层深层神经透视雷达，扫描每层第一主导宗门(抓内鬼)")
    print("   /show_k            ➔ 查看当前 28 层大小核算力分布与微专家并发规模")
    print("\n🎛️ [动态调频与弹性开核]")
    print("   /set_k <层> <核数> ➔ 单层动态开核 (自动使 Graph 失效并重新捕获)")
    print("   /set_k_all <核数>  ➔ 全局一键调频 (例: /set_k_all 1 极速狂飙, /set_k_all 4 极限算力)")
    print("\n🔒 [脑叶手术与神经禁闭]")
    print("   /cage <宗门号>     ➔ 全局封印宗门 (权重置零 + 路由打入冷宫，即时生效)")
    print("   /free <宗门号>     ➔ 刑满释放宗门 (恢复权重与路由，即时生效)")
    print("   /snipe <层> <宗门> ➔ 单层定点击毙 (单层切除 + 路由屏蔽)")
    print("\n🔌 [赛博义体与热插拔]")
    print("   /plug <卡带.pt> [槽]➔ 毫秒级热插拔技能卡带 (默认插槽 #16)")
    print("\n🧹 [系统与会话管理]")
    print("   /help (或 /h)      ➔ 呼出本指南手册")
    print("   clear              ➔ 清空当前对话上下文记忆")
    print("   exit (或 quit)     ➔ 退出终端")
    print("─" * 70)
    print("📋 [20 宗门编号速查表 (Cluster IDs)]:")
    print("   💻 代码宗门: #00:Algo | #01:DS | #02:Debug | #03:Arch")
    print("   📐 数学宗门: #04:Algebra | #05:Geo | #06:Prob | #07:Arith")
    print("   🔬 科学宗门: #08:Physics | #09:Chem | #10:Biology | #11:Astronomy")
    print("   🏛️ 人文宗门: #12:Rhetoric | #13:Philosophy | #14:Summary | #15:Chat")
    print("   🌟 特区插槽: #16:Rules | #17:Knowledge | #18:Persona | #19:Logic")
    print("═" * 70 + "\n")


def get_multiline_input():
    print("\n👤 You (支持多行粘贴，输入完成后按 Ctrl+D 或另起一行输 'EOF' 提交):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        except KeyboardInterrupt:
            return None
        if line.strip().upper() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    model_id = "Qwen/Qwen3-0.6B"
    weights_path = "myriad_moe_hierarchical_weights.pt"
    if not os.path.exists(weights_path):
        weights_path = "myriad_moe_25k_weights.pt"

    assert os.path.exists(weights_path), f"找不到权重文件，请先执行训练脚本产出权重！"

    print("=" * 70)
    print("🚀 正在唤醒【Myriad-MoE: 极限性能 25,200 专家宇宙终端】...")
    print(f"📦 挂载权重路径: {weights_path}")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 🌟 P2: 优先启用 flex_attention 规避 Blackwell (sm120) 架构上的 Ampere 回退问题
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda:0",
            attn_implementation="flex_attention"
        )
        print("⚡ [P2 优化生效] 成功启用 FlexAttention 硬件注意力加速！")
    except Exception as e:
        print(f"⚠️ [FlexAttention] 无法加载 ({e})，安全回退至默认注意力实现...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="cuda:0"
        )

    hidden_dim = model.config.hidden_size

    # 🌟 28 层异构金字塔开核初始表
    LAYER_TOP_K = [
        1, 1, 1, 1, 2, 2,          # Layer 00~05 (浅层)
        3, 3, 3, 4, 4, 4, 3, 3,    # Layer 06~13 (中浅层核心逻辑)
        3, 3, 4, 4, 3, 3, 2, 2,    # Layer 14~21 (中深层语义整合)
        1, 1, 1, 1, 1, 1           # Layer 22~27 (深层语言收敛)
    ]

    for i, layer in enumerate(model.model.layers):
        layer_k = LAYER_TOP_K[i] if i < len(LAYER_TOP_K) else 2
        layer.mlp = MyriadInferenceWrapper(layer.mlp,
                                           hidden_dim=hidden_dim,
                                           num_clusters=20,
                                           experts_per_cluster=45,
                                           micro_rank=16,
                                           macro_rank=64,
                                           device="cuda:0",
                                           dtype=dtype,
                                           top_k=layer_k)

    print(f"[*] 正在挂载 25,200 个微专家到显存...")
    saved = torch.load(weights_path, map_location="cpu")

    for i, layer in enumerate(model.model.layers):
        layer.mlp.sci_lora_A.load_state_dict(saved[f"layer_{i}_sci_lora_A"])
        layer.mlp.sci_lora_B.load_state_dict(saved[f"layer_{i}_sci_lora_B"])
        layer.mlp.router_big.load_state_dict(saved[f"layer_{i}_router_big"])
        layer.mlp.router_cluster.load_state_dict(saved[f"layer_{i}_router_cluster"])

        layer.mlp.lora_A_gpu = saved[f"layer_{i}_lora_A"].to("cuda:0", dtype=dtype)
        layer.mlp.lora_B_gpu = saved[f"layer_{i}_lora_B"].to("cuda:0", dtype=dtype)
        
        # 🌟 P0: 挂载完成后一次性建立 storage 共享视图
        layer.mlp.setup_flat_views()

    # 🌟 P1: 创建统一的 StaticCache 与 CUDA Graph 运行管理器
    MAXLEN = 2048
    cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAXLEN, device="cuda:0", dtype=dtype)
    graph_runner = CUDAGraphRunner(model, max_cache_len=MAXLEN, device="cuda:0")

    print("\n✅ 两万五千微专家已全员就位 (Flat BMM 视图 + CUDA Graph 引擎就绪)！")
    print("👉 提示：输入 '/help' 查看指令指南，输入 'clear' 重置记忆，输入 'exit' 退出\n")

    messages = [{"role": "system", "content": "You are a master of all domains with 25,200 modular micro-experts."}]
    caged_storage = {}
    eos_token_ids = set([tokenizer.eos_token_id, 151645])  # 包含 <|im_end|>

    while True:
        user_input = get_multiline_input()
        if user_input is None:
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": "You are a master of all domains with 25,200 modular micro-experts."}]
            print("🧹 记忆已重置。")
            continue

        if user_input.strip().lower() in ["/help", "/h", "help"]:
            print_help_menu()
            continue

        if user_input.strip() == "/show_k":
            print("\n" + "═" * 70)
            print("🎛️【28 层神经大小核算力配置全息表】")
            print("─" * 70)
            for idx, layer in enumerate(model.model.layers):
                k = layer.mlp.top_k
                print(f"   - Layer {idx:02d} : 开启 {k} 个宗门小核 [{'🔲'*k:<10}] ({k*45:3d} 个微专家并发)")
            print("═" * 70 + "\n")
            continue

        # 🌟 调频指令自动重置 Graph (因为 Tensor Shape 发生变化)
        if user_input.startswith("/set_k "):
            parts = user_input.split()
            if len(parts) < 3:
                print("⚠️ 用法格式: /set_k <层数> <开核数> (例如: /set_k 9 4)")
                continue
            l_idx = int(parts[1])
            if 0 <= l_idx < len(model.model.layers):
                new_k = max(1, min(int(parts[2]), 10))
                model.model.layers[l_idx].mlp.top_k = new_k
                graph_runner.invalidate()
                print(f"⚡ [调频成功] 第 {l_idx:02d} 层小核数已调整为: {new_k} 核！立即生效！\n")
            else:
                print(f"⚠️ 层数无效！")
            continue

        if user_input.startswith("/set_k_all "):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法格式: /set_k_all <小核数>")
                continue
            new_k = max(1, min(int(parts[1]), 10))
            for layer in model.model.layers:
                layer.mlp.top_k = new_k
            graph_runner.invalidate()
            print(f"⚡ [全局调频] 28 层已统一设置为每层: {new_k} 个宗门小核！立即生效！\n")
            continue

        if user_input.strip() == "/catch":
            print("\n" + "═" * 70)
            print("🚨【赛博内鬼缉捕雷达：28 层深层神经透视】")
            print("─" * 70)
            print("🔬 各深度神经层【第一主导宗门】分布扫描：")
            for idx, layer in enumerate(model.model.layers):
                counts = layer.mlp.stat_cluster_counts.cpu()
                if counts.sum() > 0:
                    top_cid = torch.argmax(counts).item()
                    cnt = counts[top_cid].item()
                    c_name = CLUSTER_NAMES[top_cid]
                    warning = " 🔥 [极度可疑]" if top_cid == 16 else ""
                    print(f"   - Layer {idx:02d} : #{top_cid:02d} [{c_name:<16}] (活跃 {cnt:3d} 拍){warning}")
                else:
                    print(f"   - Layer {idx:02d} : 暂无前向激活数据 (请先跑一次推理)")
            print("═" * 70 + "\n")
            continue

        # 🌟 原地写入自动穿透到 Af / Bf 与 CUDA Graph，无需失效重捕获
        if user_input.startswith("/cage"):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法: /cage <宗门号>")
                continue
            cid = int(parts[1])
            if cid not in caged_storage:
                caged_storage[cid] = []
                for layer in model.model.layers:
                    caged_storage[cid].append(layer.mlp.lora_B_gpu[cid].clone())
                    layer.mlp.lora_B_gpu[cid].zero_()
                    layer.mlp.cluster_bias[cid] = -1e4
                print(f"🔒 [已关禁闭] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 权重已清零 + 路由已被打入冷宫！立即生效！\n")
            continue

        if user_input.startswith("/free"):
            parts = user_input.split()
            cid = int(parts[1])
            if cid in caged_storage:
                for idx, layer in enumerate(model.model.layers):
                    layer.mlp.lora_B_gpu[cid].copy_(caged_storage[cid][idx])
                    layer.mlp.cluster_bias[cid] = 0.0
                del caged_storage[cid]
                print(f"🔓 [刑满释放] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 已恢复全额算力与路由权限！\n")
            continue

        if user_input.startswith("/snipe"):
            parts = user_input.split()
            if len(parts) < 3:
                print("⚠️ 用法: /snipe <层数> <宗门号>")
                continue
            l_idx = int(parts[1])
            cid = int(parts[2])
            model.model.layers[l_idx].mlp.lora_B_gpu[cid].zero_()
            model.model.layers[l_idx].mlp.cluster_bias[cid] = -1e4
            print(f"🎯 [狙击完毕] 第 {l_idx} 层的 #{cid} 宗门权重已清零且路由已封杀！\n")
            continue

        if user_input.startswith("/plug"):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法: /plug <卡带文件名.pt> [插槽号]")
                continue
            cart_file = parts[1]
            target_slot = int(parts[2]) if len(parts) > 2 else 16
            if not os.path.exists(cart_file):
                print(f"❌ 找不到卡带文件: {cart_file}")
                continue
            t_plug = time.perf_counter()
            cart = torch.load(cart_file, map_location="cpu")
            cart_name = cart.get("name", cart_file)
            for i, layer in enumerate(model.model.layers):
                layer_data = cart["layers"][i]
                layer.mlp.lora_A_gpu[target_slot].copy_(layer_data["lora_A"].to("cuda:0"))
                layer.mlp.lora_B_gpu[target_slot].copy_(layer_data["lora_B"].to("cuda:0"))
                layer.mlp.router_cluster.weight.data[target_slot].copy_(layer_data["router_vec"].to("cuda:0"))
                layer.mlp.cluster_bias[target_slot] = 0.0
            elapsed_ms = (time.perf_counter() - t_plug) * 1000
            print(f"\n⚡ [热插拔成功] 技能卡带《{cart_name}》已植入插槽 #{target_slot:02d} (耗时 {elapsed_ms:.2f} ms)！\n")
            continue

        # ═══════════════════════════════════════════════════════════════
        # 🌟 推理执行主干：Eager Prefill + CUDA Graph Replay Decode
        # ═══════════════════════════════════════════════════════════════
        for layer in model.model.layers:
            layer.mlp.reset_stats()

        messages.append({"role": "user", "content": user_input})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")
        prompt_ids = inputs.input_ids
        prompt_len = prompt_ids.shape[1]

        if prompt_len >= MAXLEN - 64:
            print(f"⚠️ Prompt 长度 ({prompt_len}) 接近上限 ({MAXLEN})，请先输入 'clear' 清理上下文！")
            continue

        print("\n🤖 Assistant: ", end="", flush=True)
        generated_tokens = []
        accumulated_text = ""
        t0 = time.perf_counter()

        with torch.inference_mode():
            # 1. 重置静态缓存（保持指针物理地址不变，确保 Graph 跨轮复用）
            cache.reset()

            # 2. Prefill 阶段 (Eager 执行)
            cache_pos = torch.arange(0, prompt_len, device="cuda:0")
            prefill_out = model(
                input_ids=prompt_ids,
                position_ids=cache_pos.unsqueeze(0),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos
            )

            # 采样第一个 Token
            first_tok_logits = prefill_out.logits[0, -1, :]
            next_tok = sample_next_token(first_tok_logits, generated_tokens)
            next_tok_id = next_tok.item()
            generated_tokens.append(next_tok_id)

            cur_word = tokenizer.decode([next_tok_id], skip_special_tokens=True)
            print(cur_word, end="", flush=True)
            accumulated_text += cur_word

            # 3. Decode 阶段 (CUDA Graph 极速重放循环)
            cur_pos = prompt_len
            max_gen = 2048 - prompt_len - 1

            for _ in range(max_gen):
                if next_tok_id in eos_token_ids or cur_pos >= MAXLEN - 1:
                    break

                # 首次或调频后重新捕获
                if graph_runner.graph is None:
                    graph_runner.capture_decode_graph(cache, next_tok_id, cur_pos)

                # 纯 GPU 重放，Dispatch 延迟趋近于 0
                step_logits = graph_runner.replay_step(next_tok_id, cur_pos)

                # 图外轻量采样
                next_tok = sample_next_token(step_logits, generated_tokens)
                next_tok_id = next_tok.item()
                cur_pos += 1

                if next_tok_id in eos_token_ids:
                    break

                generated_tokens.append(next_tok_id)
                cur_word = tokenizer.decode([next_tok_id], skip_special_tokens=True)
                print(cur_word, end="", flush=True)
                accumulated_text += cur_word

        elapsed_sec = time.perf_counter() - t0
        gen_tokens = len(generated_tokens)
        speed = gen_tokens / elapsed_sec if elapsed_sec > 0 else 0
        print(f"\n\n⚡ 极速: {speed:.1f} tokens/s (共 {gen_tokens} 字, 耗时 {elapsed_sec*1000:.0f} ms)")

        show_myriad_dashboard(model)
        messages.append({"role": "assistant", "content": accumulated_text})


if __name__ == "__main__":
    main()

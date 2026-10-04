import os
import sys
import time
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

# 🌟 20 宗门定义（含 4 大预留插槽）
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
    "Custom_Rules",      # #16: 业务/工坊硬规则插槽 (预备替换)
    "Custom_Knowledge",  # #17: 私有知识库插槽
    "Custom_Persona",    # #18: 专属人设/角色插槽
    "Custom_Logic"       # #19: 专属推理/任务插槽
]

# ═══════════════════════════════════════════════════════════════
# 🌟 极速弹性推理包装层：支持任意层按需动态开核 (Dynamic Top-K)
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
                 reside_on_gpu=True,
                 top_k=2):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.micro_rank = micro_rank
        self.macro_rank = macro_rank
        self.reside_on_gpu = reside_on_gpu
        self.top_k = top_k  # 当前层开启的小核宗门数

        # 🌟 微专家执行缩放系数 (保持原设计 0.025)
        self.micro_scale = 0.025

        # [L0 底座]: 唯一底盘 MLP
        self.base_mlp = original_mlp.to(device)

        # [L1 理科大核]: 高阶 LoRA (macro_rank=64)
        self.sci_lora_A = nn.Linear(hidden_dim, macro_rank, bias=False, device=device, dtype=dtype)
        self.sci_lora_B = nn.Linear(macro_rank, hidden_dim, bias=False, device=device, dtype=dtype)

        # 路由器
        self.router_big = nn.Linear(hidden_dim, 2, bias=False, device=device, dtype=dtype)
        self.router_cluster = nn.Linear(hidden_dim, num_clusters, bias=False, device=device, dtype=dtype)

        # 专家权重池引用
        self.lora_A_gpu = None
        self.lora_B_gpu = None
        self.lora_A_cpu = None
        self.lora_B_cpu = None

        # 🌟 扁平化「视图」(共享同一物理显存 storage，零额外显存占用)
        self.Af = None
        self.Bf = None

        # GPU 原地计数器 (消灭 Python 循环内 .item() 同步阻断)
        self.register_buffer("stat_arts_weight", torch.zeros(1, device=device, dtype=torch.float32))
        self.register_buffer("stat_sci_weight", torch.zeros(1, device=device, dtype=torch.float32))
        self.register_buffer("stat_cluster_counts", torch.zeros(num_clusters, device=device, dtype=torch.int32))

        # 禁闭电磁屏蔽偏置网 (平时为 0，关禁闭/狙击时被打入 -1e4 冷宫)
        self.register_buffer("cluster_bias", torch.zeros(num_clusters, device=device, dtype=dtype))

        # 免分配的辅助常量
        self.register_buffer("expert_ar", torch.arange(experts_per_cluster, device=device), persistent=False)
        self.register_buffer("stat_ones", torch.ones(num_clusters, device=device, dtype=torch.int32), persistent=False)

    def mount_experts(self, lora_A, lora_B):
        """挂载权重并建立零拷贝扁平化 view"""
        self.lora_A_gpu = lora_A
        self.lora_B_gpu = lora_B
        flat = self.num_clusters * self.experts_per_cluster
        self.Af = lora_A.reshape(flat, self.micro_rank * self.hidden_dim)
        self.Bf = lora_B.reshape(flat, self.hidden_dim * self.micro_rank)
        assert self.Af.data_ptr() == self.lora_A_gpu.data_ptr(), "Af 必须与 lora_A_gpu 共享底层显存物理地址！"
        assert self.Bf.data_ptr() == self.lora_B_gpu.data_ptr(), "Bf 必须与 lora_B_gpu 共享底层显存物理地址！"
        return self

    def set_top_k(self, k):
        self.top_k = max(1, min(int(k), self.num_clusters))

    def reset_stats(self):
        self.stat_arts_weight.zero_()
        self.stat_sci_weight.zero_()
        self.stat_cluster_counts.zero_()

    def _micro_forward(self, x, topk_scores, topk_clusters):
        b, s, d = x.shape
        E = self.experts_per_cluster
        R = self.micro_rank
        k = topk_clusters.shape[-1]
        scale = 1.0 / E

        if self.reside_on_gpu and self.Af is not None and b == 1:
            if s == 1:
                # ---- Decode: 逐字吐字，一次 gather 全部命中专家，两次 GEMM
                idx = (topk_clusters.view(-1, 1) * E + self.expert_ar).view(-1)      # [k*E]
                As = self.Af.index_select(0, idx).view(k, E * R, d)                 # [k, 720, D]
                h = torch.matmul(As, x.view(d)).view(k * E, R)                      # [k*E, R]
                Bs = self.Bf.index_select(0, idx).view(k * E, d, R)                 # [k*E, D, R]
                o = torch.bmm(Bs, h.unsqueeze(-1)).view(k, E, d).sum(1) * scale      # [k, D]
                out = (o.float() * topk_scores.view(k, 1)).sum(0)                   # [D]
                return out.view(1, 1, d).to(x.dtype)

            # ---- Prefill: 一次 gather + 两个批量 einsum
            sel = topk_clusters.view(-1)                                            # [k]
            Asel = self.lora_A_gpu.index_select(0, sel).view(k, E, R, d)
            Bsel = self.lora_B_gpu.index_select(0, sel).view(k, E, d, R)
            h = torch.einsum('bsd,kerd->bsker', x, Asel)                            # [b,s,k,E,R]
            o = torch.einsum('bsker,kedr->bskd', h, Bsel)                           # [b,s,k,D]
            out = (o.float() * topk_scores.view(b, 1, k, 1)).sum(2) * scale          # [b,s,D]
            return out.view(b, s, d).to(x.dtype)

        # ---- CPU 常驻回退保底路径
        micro_out = 0
        xv = x.view(d, 1)
        for kid in range(k):
            cid = topk_clusters[0, 0, kid]
            if self.reside_on_gpu:
                A = self.lora_A_gpu[cid]
                B = self.lora_B_gpu[cid]
            else:
                A = self.lora_A_cpu[cid.item()].to(self.device, non_blocking=True)
                B = self.lora_B_cpu[cid.item()].to(self.device, non_blocking=True)
            if s == 1:
                h = torch.matmul(A, xv)
                out_k = (torch.sum(torch.matmul(B, h), dim=0) * scale).view(1, 1, d)
            else:
                h = torch.einsum('bsd,erd->bser', x, A)
                out_k = torch.einsum('bser,edr->bsd', h, B) * scale
            micro_out = micro_out + topk_scores[0, 0, kid] * out_k
        return micro_out

    def forward(self, x):
        current_token = x[:, -1:, :]

        # 1. 双大核文理动态调度 (纯 GPU 原地无锁累加)
        logits_big = self.router_big(current_token)
        w_big = torch.softmax(logits_big, dim=-1)
        self.stat_arts_weight += w_big[0, 0, 0]
        self.stat_sci_weight += w_big[0, 0, 1]

        base_out = self.base_mlp(x)
        sci_delta = 0.1 * self.sci_lora_B(self.sci_lora_A(x))
        big_out = base_out + (w_big[..., 1:2] * sci_delta)

        # 2. 动态路由：根据本层的 self.top_k 自由开核
        logits_cluster = self.router_cluster(current_token) + self.cluster_bias
        w_cluster = torch.softmax(logits_cluster, dim=-1)

        cur_k = min(self.top_k, self.num_clusters)
        topk_scores, topk_clusters = torch.topk(w_cluster, k=cur_k, dim=-1)

        # GPU 原地记录热度 (1 次 scatter_add_)
        self.stat_cluster_counts.scatter_add_(0, topk_clusters.reshape(-1), self.stat_ones[:cur_k])

        # 3. 极速执行区
        micro_out = self._micro_forward(x, topk_scores, topk_clusters)

        return big_out + self.micro_scale * micro_out


# ═══════════════════════════════════════════════════════════════
# 🌟 CUDA Graph 极速解码器 (带 Cache 分级升档与 Greedy 保护)
# ═══════════════════════════════════════════════════════════════
class GraphedDecoder:
    CACHE_BUCKETS = (512, 768, 1024, 1536, 2048, 3072, 4096, 8192, 16384, 32768)

    def __init__(self, model, device, dtype, enabled=True, verbose=True):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.enabled = enabled
        self.verbose = verbose
        self.cache = None
        self.cache_len = 0
        self.graph = None
        self.graph_out = None
        self.graph_len = 0
        self.step_tok = None

    def invalidate(self, reason=""):
        self.graph = None
        if self.verbose and reason:
            print(f"♻️  [CUDA Graph 已失效] {reason} → 下一轮将自动重新捕获")

    def _pick_bucket(self, need):
        for b in self.CACHE_BUCKETS:
            if need <= b:
                return b
        return need

    def _ensure_cache(self, need):
        want = self._pick_bucket(need)
        if self.cache is not None and self.cache_len >= want:
            return
        self.cache = StaticCache(config=self.model.config, max_batch_size=1,
                                 max_cache_len=want, device=self.device, dtype=self.dtype)
        self.cache_len = want
        self.graph = None  # 缓冲区地址已变，旧图作废
        if self.verbose:
            print(f"📐 [KV Cache 扩容] 容量 {want} tokens (StaticCache)")

    def _capture(self):
        model = self.model
        self.step_tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        t0 = time.perf_counter()

        # 1) side stream 预热
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                out = model(input_ids=self.step_tok, past_key_values=self.cache, use_cache=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        # 2) 录制整步 decode 硬件图
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = model(input_ids=self.step_tok, past_key_values=self.cache, use_cache=True)

        self.graph = graph
        self.graph_out = out
        self.graph_len = self.cache_len
        self.cache.reset()
        if self.verbose:
            print(f"⚡ [CUDA Graph 捕获完成] {(time.perf_counter() - t0) * 1000:.0f} ms "
                  f"(cache={self.cache_len}) — 之后每字仅需一次 replay()")

    def _sample(self, logits, hist, temperature, top_p, repetition_penalty):
        scores = logits
        if repetition_penalty is not None and repetition_penalty != 1.0:
            sc = torch.gather(scores, 1, hist)
            sc = torch.where(sc < 0, sc * repetition_penalty, sc / repetition_penalty)
            scores = scores.scatter(1, hist, sc)

        # 🌟 修复点 1：增加 temperature <= 0 时的纯贪婪解码保护，彻底防止除零 NaN
        if temperature is None or temperature <= 1e-5:
            return torch.argmax(scores, dim=-1, keepdim=True)

        scores = scores / temperature
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, dim=-1)
            probs = torch.softmax(sorted_scores, dim=-1)
            remove = (torch.cumsum(probs, dim=-1) - probs) > top_p
            remove[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(remove, torch.finfo(scores.dtype).min)
            scores = torch.full_like(scores, torch.finfo(scores.dtype).min).scatter(1, sorted_idx, sorted_scores)
        return torch.multinomial(torch.softmax(scores, dim=-1), 1)

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, eos_token_ids,
                 temperature=0.7, top_p=0.9, repetition_penalty=1.0):
        prompt_len = input_ids.shape[1]
        eos_token_ids = set(eos_token_ids)

        if not self.enabled:
            yield from self._generate_eager(input_ids, max_new_tokens, eos_token_ids,
                                            temperature, top_p, repetition_penalty)
            return

        self._ensure_cache(prompt_len + max_new_tokens + 1)
        if self.graph is None or self.graph_len != self.cache_len:
            self._capture()

        # 干净的真实 prefill
        self.cache.reset()
        out = self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True)

        logits = out.logits[:, -1, :].float()
        hist = input_ids
        produced = 0
        while produced < max_new_tokens:
            nxt = self._sample(logits, hist, temperature, top_p, repetition_penalty)
            tid = int(nxt)
            if tid in eos_token_ids:
                return
            yield tid
            produced += 1
            hist = torch.cat([hist, nxt], dim=1)
            self.step_tok.copy_(nxt)
            self.graph.replay()
            logits = self.graph_out.logits[:, -1, :].float()

    @torch.inference_mode()
    def _generate_eager(self, input_ids, max_new_tokens, eos_token_ids,
                        temperature, top_p, repetition_penalty):
        from transformers import DynamicCache
        cache = DynamicCache()
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :].float()
        hist = input_ids
        for _ in range(max_new_tokens):
            nxt = self._sample(logits, hist, temperature, top_p, repetition_penalty)
            tid = int(nxt)
            if tid in eos_token_ids:
                return
            yield tid
            out = self.model(input_ids=nxt, past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1, :].float()
            hist = torch.cat([hist, nxt], dim=1)


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
    print("   /set_k <层> <核数> ➔ 单层动态开核 (例: /set_k 9 4 把第 9 层开到 4 核)")
    print("   /set_k_all <核数>  ➔ 全局一键调频 (例: /set_k_all 1 极速狂飙, /set_k_all 4 极限算力)")
    print("   ⚠️ 改开核数会改变张量形状，CUDA Graph 会自动重新捕获 (~0.3s)")
    print("\n⚡ [引擎开关与参数]")
    print("   /graph             ➔ 切换 CUDA Graph 极速解码 / 普通 eager 解码 (现场对比)")
    print("   /maxlen <n>        ➔ 单次回复最大 token 数 (默认 512，StaticCache 自动升档)")
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
    print("🚀 正在唤醒【Myriad-MoE: 极速 25,200 专家宇宙终端】...")
    print(f"📦 挂载权重路径: {weights_path}")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 🌟 5090 Blackwell 底层加速配置
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0",
                                                 attn_implementation="sdpa")
    model.eval()
    hidden_dim = model.config.hidden_size

    USE_GPU_RESIDENT = True

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
                                           reside_on_gpu=USE_GPU_RESIDENT,
                                           top_k=layer_k)

    print(f"[*] 正在挂载 25,200 个微专家...")
    saved = torch.load(weights_path, map_location="cpu")

    for i, layer in enumerate(model.model.layers):
        layer.mlp.sci_lora_A.load_state_dict(saved[f"layer_{i}_sci_lora_A"])
        layer.mlp.sci_lora_B.load_state_dict(saved[f"layer_{i}_sci_lora_B"])
        layer.mlp.router_big.load_state_dict(saved[f"layer_{i}_router_big"])
        layer.mlp.router_cluster.load_state_dict(saved[f"layer_{i}_router_cluster"])

        if USE_GPU_RESIDENT:
            layer.mlp.mount_experts(saved[f"layer_{i}_lora_A"].to("cuda:0", dtype=dtype).contiguous(),
                                    saved[f"layer_{i}_lora_B"].to("cuda:0", dtype=dtype).contiguous())
        else:
            layer.mlp.lora_A_cpu = saved[f"layer_{i}_lora_A"].pin_memory()
            layer.mlp.lora_B_cpu = saved[f"layer_{i}_lora_B"].pin_memory()
    del saved

    print("\n✅ 两万五千微专家已全员就位 (GPU 纯显存零开销调度已就绪)！")
    print("👉 提示：随时输入 '/help' 查看指令指南，输入 'clear' 重置记忆，输入 'exit' 退出\n")

    decoder = GraphedDecoder(model, "cuda:0", dtype,
                             enabled=USE_GPU_RESIDENT,
                             verbose=True)
    max_new_tokens = 512

    messages = [{"role": "system", "content": "You are a master of all domains with 25,200 modular micro-experts."}]
    caged_storage = {}

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

        if user_input.startswith("/set_k "):
            parts = user_input.split()
            if len(parts) < 3:
                print("⚠️ 用法格式: /set_k <层数> <开核数> (例如: /set_k 9 4)")
                continue
            l_idx = int(parts[1])
            if 0 <= l_idx < len(model.model.layers):
                new_k = max(1, min(int(parts[2]), 10))
                model.model.layers[l_idx].mlp.top_k = new_k
                decoder.invalidate(f"第 {l_idx:02d} 层开核数 -> {new_k}")
                print(f"⚡ [调频成功] 第 {l_idx:02d} 层小核数已调整为: {new_k} 核 (共 {new_k*45} 个微专家)！立即生效！\n")
            else:
                print(f"⚠️ 层数无效，请输入 0 ~ {len(model.model.layers)-1} 之间的层号！")
            continue

        if user_input.startswith("/set_k_all "):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法格式: /set_k_all <小核数> (例如: /set_k_all 1 极速狂飙, /set_k_all 4 极限算力)")
                continue
            new_k = max(1, min(int(parts[1]), 10))
            for layer in model.model.layers:
                layer.mlp.top_k = new_k
            decoder.invalidate(f"全局开核数 -> {new_k}")
            print(f"⚡ [全局调频] 28 层已统一设置为每层: {new_k} 个宗门小核 (全网每步激活 {new_k*45*28:,} 个微专家)！立即生效！\n")
            continue

        if user_input.strip() == "/graph":
            decoder.enabled = not decoder.enabled
            decoder.invalidate("手动切换引擎")
            state = "CUDA Graph 极速解码 (已启用)" if decoder.enabled else "普通 eager 解码 (已关闭)"
            print(f"🔧 [引擎切换] {state}\n")
            continue

        if user_input.strip().startswith("/maxlen"):
            parts = user_input.split()
            if len(parts) < 2:
                print(f"⚠️ 用法: /maxlen <token数> (当前 {max_new_tokens})")
                continue
            new_ml = max(64, min(int(parts[1]), 8192))
            print(f"📏 [回复上限] {max_new_tokens} ➔ {new_ml} tokens (KV Cache 会自动分档升档)\n")
            max_new_tokens = new_ml
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

        if user_input.startswith("/cage"):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法: /cage <宗门号> (例如: /cage 16)")
                continue
            cid = int(parts[1])
            if cid not in caged_storage:
                caged_storage[cid] = []
                for layer in model.model.layers:
                    target_b = layer.mlp.lora_B_gpu if USE_GPU_RESIDENT else layer.mlp.lora_B_cpu
                    caged_storage[cid].append(target_b[cid].clone())
                    target_b[cid].zero_()
                    layer.mlp.cluster_bias[cid] = -1e4
                print(f"🔒 [已关禁闭] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 权重已清零 + 路由已被打入冷宫！立即生效！\n")
            else:
                print(f"⚠️ 宗门 #{cid:02d} 已经在禁闭室了！")
            continue

        if user_input.startswith("/free"):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法: /free <宗门号>")
                continue
            cid = int(parts[1])
            if cid in caged_storage:
                for idx, layer in enumerate(model.model.layers):
                    target_b = layer.mlp.lora_B_gpu if USE_GPU_RESIDENT else layer.mlp.lora_B_cpu
                    target_b[cid].copy_(caged_storage[cid][idx])
                    layer.mlp.cluster_bias[cid] = 0.0
                del caged_storage[cid]
                print(f"🔓 [刑满释放] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 已恢复全额算力与路由权限！\n")
            else:
                print(f"⚠️ 宗门 #{cid:02d} 并没有被关押！")
            continue

        if user_input.startswith("/snipe"):
            parts = user_input.split()
            if len(parts) < 3:
                print("⚠️ 用法格式: /snipe <层数> <宗门号> (例如: /snipe 21 16)")
                continue
            l_idx = int(parts[1])
            cid = int(parts[2])
            target_b = model.model.layers[l_idx].mlp.lora_B_gpu if USE_GPU_RESIDENT else model.model.layers[l_idx].mlp.lora_B_cpu
            target_b[cid].zero_()
            model.model.layers[l_idx].mlp.cluster_bias[cid] = -1e4
            print(f"🎯 [狙击完毕] 第 {l_idx} 层的 #{cid} 宗门权重已被原地清零，且本层路由已封杀！\n")
            continue

        if user_input.startswith("/plug"):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法格式: /plug <卡带文件名.pt> [可选插槽编号, 默认16]")
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
                if USE_GPU_RESIDENT:
                    layer.mlp.lora_A_gpu[target_slot].copy_(layer_data["lora_A"].to("cuda:0"))
                    layer.mlp.lora_B_gpu[target_slot].copy_(layer_data["lora_B"].to("cuda:0"))
                else:
                    layer.mlp.lora_A_cpu[target_slot].copy_(layer_data["lora_A"])
                    layer.mlp.lora_B_cpu[target_slot].copy_(layer_data["lora_B"])
                layer.mlp.router_cluster.weight.data[target_slot].copy_(layer_data["router_vec"].to("cuda:0"))
                layer.mlp.cluster_bias[target_slot] = 0.0

            elapsed_ms = (time.perf_counter() - t_plug) * 1000
            print(f"\n⚡ [热插拔成功] 技能卡带《{cart_name}》已就地植入插槽 #{target_slot:02d} (耗时 {elapsed_ms:.2f} ms)！\n")
            continue

        # 统计重置
        for layer in model.model.layers:
            layer.mlp.reset_stats()

        messages.append({"role": "user", "content": user_input})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to("cuda:0")

        eos_ids = [tokenizer.eos_token_id, 151645]

        print("\n🤖 Assistant: ", end="", flush=True)

        t0 = time.perf_counter()
        
        # 🌟 修复：用 generated_token_ids 保存完整序列，由 tokenizer 自动处理跨步多字节拼接
        generated_token_ids = []
        prev_text = ""
        for tid in decoder.generate(input_ids,
                                    max_new_tokens=max_new_tokens,
                                    eos_token_ids=eos_ids,
                                    temperature=0.7,
                                    top_p=0.9,
                                    repetition_penalty=1.15):
            generated_token_ids.append(tid)
            # 整体解码，遇到半个汉字时会自动等待下一个字节，绝对不会输出 
            current_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
            if len(current_text) > len(prev_text):
                print(current_text[len(prev_text):], end="", flush=True)
                prev_text = current_text

        accumulated_text = prev_text

        elapsed_sec = time.perf_counter() - t0
        gen_tokens = len(generated_token_ids)
        speed = gen_tokens / elapsed_sec if elapsed_sec > 0 else 0
        print(f"\n\n⚡ 速度: {speed:.1f} tokens/s (共 {gen_tokens} 字, 耗时 {elapsed_sec*1000:.0f} ms)")

        show_myriad_dashboard(model)
        messages.append({"role": "assistant", "content": accumulated_text})


if __name__ == "__main__":
    main()

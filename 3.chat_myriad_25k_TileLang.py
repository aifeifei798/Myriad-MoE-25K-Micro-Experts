import os
import sys
import time
import copy
from threading import Thread
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import tilelang
import tilelang.language as T

# 🌟 扩展至 20 宗门定义（含 4 大预留插槽）
CLUSTER_NAMES = [
    # 0~3: 代码宗门
    "Code_Algo",
    "Code_DS",
    "Code_Debug",
    "Code_Arch",
    # 4~7: 数学宗门
    "Math_Algebra",
    "Math_Geo",
    "Math_Prob",
    "Math_Arith",
    # 8~11: 科学宗门
    "Sci_Physics",
    "Sci_Chem",
    "Sci_Biology",
    "Sci_Astronomy",
    # 12~15: 人文宗门
    "Arts_Rhetoric",
    "Arts_Philosophy",
    "Arts_Summary",
    "Arts_Chat",
    # 🌟 16~19: 自定义热插拔特区
    "Custom_Rules",  # #16: 业务/工坊硬规则插槽 (预备替换)
    "Custom_Knowledge",  # #17: 私有知识库插槽
    "Custom_Persona",  # #18: 专属人设/角色插槽
    "Custom_Logic"  # #19: 专属推理/任务插槽
]

# ═══════════════════════════════════════════════════════════════
# 🌟 修复版 TileLang 融合算子
# 计算: Out[M, D] = (1/E) * sum_e ( X[M, D] @ A[e, R, D]^T ) @ B[e, D, R]^T
# ═══════════════════════════════════════════════════════════════
@tilelang.jit
def fused_multi_lora_kernel(
    X,  # [M, D] bf16
    A,  # [E, R, D] bf16
    B,  # [E, D, R] bf16
    Out,  # [M, D] fp32 (必须先清零, 专家维度用 atomic_add 归约)
    block_M: int = 16,
    block_D: int = 64):
    M, D, E, R = T.const("M, D, E, R")

    X: T.Tensor((M, D), T.bfloat16)
    A: T.Tensor((E, R, D), T.bfloat16)
    B: T.Tensor((E, D, R), T.bfloat16)
    Out: T.Tensor((M, D), T.float32)

    # 🌟 修正 1: 专家维 E 提到 grid 的第 3 维做并行 (原写法 for e in T.Serial(E)
    #            把 45 个专家串行跑在一个 CTA 里, 慢且占据全部时间)
    # 🌟 修正 2: threads=32 (原 threads=128 → 4 warps 无法覆盖 M=16,N=16 的
    #            第一个 gemm, 编译期直接 Fatal 退出)
    with T.Kernel(T.ceildiv(D, block_D),
                  T.ceildiv(M, block_M),
                  E,
                  threads=32) as (bx, by, bz):
      X_shared = T.alloc_shared((block_M, block_D), T.bfloat16)
      A_shared = T.alloc_shared((R, block_D), T.bfloat16)
      B_shared = T.alloc_shared((block_D, R), T.bfloat16)
      # 🌟 修正 3: 共享内存 / fragment 必须在循环外分配一次
      #            (原代码在 for 体内 alloc, 每轮重新分配, 布局推断失败)
      h_shared = T.alloc_shared((block_M, R), T.bfloat16)
      h_frag = T.alloc_fragment((block_M, R), T.float32)
      acc_out = T.alloc_fragment((block_M, block_D), T.float32)
      T.clear(h_frag)
      T.clear(acc_out)

      # gemm1: h[M, R] = X[M, D] @ A[bz]^T[D, R]
      for k_d in T.Pipelined(T.ceildiv(D, block_D), num_stages=3):
        T.copy(X[by * block_M, k_d * block_D], X_shared)
        # 🌟 修正 4: A[e, :, k] 切片在 dim2 留下非 1 extent, 触发
        #            "base_ranges has extra non-1 extent at dim 2"
        #            改成 T.copy(A[bz, 0, k], A_shared) 走前导维
        T.copy(A[bz, 0, k_d * block_D], A_shared)
        T.gemm(X_shared, A_shared, h_frag, transpose_B=True)

      # 🌟 修正 5: h_frag 是 fp32 fragment, B_shared 是 bf16, 混合 dtype
      #            的 gemm 会被拒绝 → 先搬进 bf16 共享内存当 gemm 输入
      for i, j in T.Parallel(block_M, R):
        h_shared[i, j] = h_frag[i, j]

      # gemm2: acc[M, D] += h[M, R] @ B[bz]^T[R, D]
      # 🌟 修正 6: B[e, bx*block_D, :] 同样有切片问题 → 用前导维
      T.copy(B[bz, bx * block_D, 0], B_shared)
      T.gemm(h_shared, B_shared, acc_out, transpose_B=True)

      # 专家维已在 grid 上并行, 用 atomic_add 汇总到 fp32 输出
      scale = 1.0 / 45.0
      for i, j in T.Parallel(block_M, block_D):
        T.atomic_add(Out[by * block_M + i, bx * block_D + j],
                     acc_out[i, j] * scale)

class MyriadInferenceWrapper(nn.Module):

    def __init__(self,
                 original_mlp,
                 hidden_dim=1024,
                 num_clusters=20,
                 experts_per_cluster=45,
                 rank=16,
                 device="cuda:0",
                 dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.rank = rank

        self.big_arts = original_mlp.to(device)
        self.big_sci = copy.deepcopy(original_mlp).to(device)

        self.router_big = nn.Linear(hidden_dim,
                                    2,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)
        # 🌟 20 宗门路由器
        self.router_cluster = nn.Linear(hidden_dim,
                                        num_clusters,
                                        bias=False,
                                        device=device,
                                        dtype=dtype)

        # 25,200 专家常驻 CPU 锁页内存 (仅 1.54 GB)
        self.lora_A_cpu = None
        self.lora_B_cpu = None
        self.intra_router = None

        self.transfer_stream = torch.cuda.Stream(device=device)

        # 统计计数
        self.total_arts_weight = 0.0
        self.total_sci_weight = 0.0
        self.cluster_counter = Counter()
        # 🌟 新增：每层深层作案记录器 (记录每次前向时到底给了谁多少推力)
        self.layer_cluster_activity = Counter()

    def reset_stats(self):
        self.total_arts_weight = 0.0
        self.total_sci_weight = 0.0
        self.cluster_counter.clear()
        self.layer_cluster_activity.clear()  # 🌟 同步清空

    def forward(self, x):
        b, s, d = x.shape
        current_token = x[:, -1:, :]

        # 1. 双大核文理动态调度 (保持不变)
        logits_big = self.router_big(current_token)
        w_big = torch.softmax(logits_big, dim=-1)
        self.total_arts_weight += w_big[0, 0, 0].item()
        self.total_sci_weight += w_big[0, 0, 1].item()

        arts_out = self.big_arts(x)
        sci_out = self.big_sci(x)
        big_out = (w_big[..., 0:1] * arts_out) + (w_big[..., 1:2] * sci_out)

        # 2. 20 宗门动态路由 (保持不变)
        logits_cluster = self.router_cluster(current_token)
        w_cluster = torch.softmax(logits_cluster, dim=-1)
        top2_scores, top2_clusters = torch.topk(w_cluster, k=2, dim=-1)
        c1 = top2_clusters[0, 0, 0].item()
        c2 = top2_clusters[0, 0, 1].item()
        self.cluster_counter[c1] += 1
        self.cluster_counter[c2] += 1
        self.layer_cluster_activity[c1] += 1

        # 3. 异步流式拉取对口宗门的专家切片 (DMA 非阻塞)
        with torch.cuda.stream(self.transfer_stream):
            A1 = self.lora_A_cpu[c1].to(self.device, non_blocking=True)
            B1 = self.lora_B_cpu[c1].to(self.device, non_blocking=True)
            A2 = self.lora_A_cpu[c2].to(self.device, non_blocking=True)
            B2 = self.lora_B_cpu[c2].to(self.device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(self.transfer_stream)

        # ═══════════════════════════════════════════════════════════════
        # 🚀【极速融合执行区】：带 16-Row 对齐 + 双轨安全容错
        # ═══════════════════════════════════════════════════════════════
        x_2d = x.view(-1, d)
        M_len = x_2d.shape[0]

        if M_len > 1:
            # 🌟 Prefill 阶段 (长 Prompt)：直接走批量 einsum
            h1 = torch.einsum('bsd,erd->bser', x, A1)
            out1 = torch.sum(torch.einsum('bser,edr->bsed', h1, B1), dim=2) / 45.0

            h2 = torch.einsum('bsd,erd->bser', x, A2)
            out2 = torch.sum(torch.einsum('bser,edr->bsed', h2, B2), dim=2) / 45.0
        else:
            # 🚀 Decode 阶段 (M=1 逐字吐字)：
            # 🌟 修正: 之前用 torch.empty 传 bf16 输出, 但算子内部用 atomic_add
            #            汇总专家维, 必须传"已清零的 fp32"累加缓冲
            try:
                # 硬件张量核要求 M>=16，Pad 15 行零对齐输入
                x_pad = torch.zeros(16, d, device=self.device, dtype=x.dtype)
                x_pad[0:1] = x_2d
                acc1 = torch.zeros(16, d, device=self.device, dtype=torch.float32)
                acc2 = torch.zeros(16, d, device=self.device, dtype=torch.float32)

                # 调用 TileLang 融合算子
                fused_multi_lora_kernel(x_pad, A1, B1, acc1)
                fused_multi_lora_kernel(x_pad, A2, B2, acc2)

                out1 = acc1[0:1].to(x.dtype).view(b, s, d)
                out2 = acc2[0:1].to(x.dtype).view(b, s, d)
            except Exception as e:
                # 🛡️ 安全兜底：走超轻量 cuBLAS Batched GEMV（无需中间 4D 显存分配，同样零卡顿）
                if not getattr(self, "_kernel_warned", False):
                    print(f"\n⚠️ [TileLang 算子不可用, 已回退到 cuBLAS 兜底路径] {type(e).__name__}: {str(e).splitlines()[0][:120]}")
                    print("   (若要修好算子, 删掉本 try/except 看完整报错)\n")
                    self._kernel_warned = True
                x_vec = x_2d.unsqueeze(-1)  # [1, 1024, 1]
                h1 = torch.matmul(A1, x_vec)  # [45, 16, 1]
                out1 = torch.matmul(B1, h1).mean(dim=0).view(b, s, d)

                h2 = torch.matmul(A2, x_vec)
                out2 = torch.matmul(B2, h2).mean(dim=0).view(b, s, d)

        micro_out = top2_scores[..., 0:1] * out1 + top2_scores[..., 1:2] * out2
        return big_out + 0.3 * micro_out


def show_myriad_dashboard(model):
    total_arts = sum(layer.mlp.total_arts_weight
                     for layer in model.model.layers)
    total_sci = sum(layer.mlp.total_sci_weight for layer in model.model.layers)
    all_big = total_arts + total_sci
    arts_pct = (total_arts / all_big * 100) if all_big > 0 else 50
    sci_pct = (total_sci / all_big * 100) if all_big > 0 else 50

    total_clusters = Counter()
    for layer in model.model.layers:
        total_clusters.update(layer.mlp.cluster_counter)

    print("\n" + "═" * 70)
    print("🌌【Myriad-MoE: 25,200 微专家宇宙全息透视 (含4大插槽)】:")
    print(f"   🏛️  文科原版大核: {arts_pct:5.1f}% [{'█'*int(arts_pct//5):<20}]")
    print(f"   🔬 理科特训大核: {sci_pct:5.1f}% [{'█'*int(sci_pct//5):<20}]")
    print("─" * 70)
    print("🪐【20 宗门活跃热力图 (Top Active Clusters)】:")
    for cid, cnt in total_clusters.most_common(6):
        c_name = CLUSTER_NAMES[cid]
        # 如果是 16~19 号插槽，打上醒目标记
        tag = " 🌟 [插槽]" if cid >= 16 else ""
        print(f"   ✨ #{cid:02d} [{c_name:<16}]: 激活 {cnt:,} 次{tag}")
    print("═" * 70)


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
    weights_path = "myriad_moe_25k_weights.pt"
    assert os.path.exists(
        weights_path), f"找不到权重文件 {weights_path}，请先执行 2.train_myriad_25k.py 训练！"

    print("=" * 70)
    print("🚀 正在唤醒【Myriad-MoE: 25,200 专家万象终端 (20 宗门版)】...")
    print("=" * 70)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")
    hidden_dim = model.config.hidden_size

    for layer in model.model.layers:
        layer.mlp = MyriadInferenceWrapper(
            layer.mlp,
            hidden_dim=hidden_dim,
            num_clusters=20,  # 🌟 20 宗门
            experts_per_cluster=45,
            rank=16,
            device="cuda:0",
            dtype=dtype)

    print(f"[*] 正在挂载 25,200 个微专家到 CPU 锁页内存 (仅吃 1.54 GB RAM)...")
    saved = torch.load(weights_path, map_location="cpu")
    for i, layer in enumerate(model.model.layers):
        layer.mlp.big_sci.load_state_dict(saved[f"layer_{i}_big_sci"])
        layer.mlp.router_big.load_state_dict(saved[f"layer_{i}_router_big"])
        layer.mlp.router_cluster.load_state_dict(
            saved[f"layer_{i}_router_cluster"])
        layer.mlp.lora_A_cpu = saved[f"layer_{i}_lora_A"].pin_memory()
        layer.mlp.lora_B_cpu = saved[f"layer_{i}_lora_B"].pin_memory()

    print("\n✅ 两万五千微专家帝国已全员就位 (含 4 大热插拔特区)！")
    print("👉 提示：输入 'clear' 重置记忆，输入 'exit' 退出\n")

    messages = [{
        "role":
        "system",
        "content":
        "You are a master of all domains with 25,200 modular micro-experts."
    }]

    # 临时备份囚禁专家的字典
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
            messages = [{
                "role":
                "system",
                "content":
                "You are a master of all domains with 25,200 modular micro-experts."
            }]
            print("🧹 记忆已重置。")
            continue

        # ═══════════════════════════════════════════════════════════════
        # 🚨 1. 抓内鬼雷达：/catch
        # ═══════════════════════════════════════════════════════════════
        if user_input.strip() == "/catch":
            print("\n" + "═" * 70)
            print("🚨【赛博内鬼缉捕雷达：28 层深层神经透视】")
            print("─" * 70)

            culprit_per_layer = []
            for idx, layer in enumerate(model.model.layers):
                if layer.mlp.layer_cluster_activity:
                    top_cid, count = layer.mlp.layer_cluster_activity.most_common(
                        1)[0]
                    culprit_per_layer.append((idx, top_cid, count))

            # 按层分组打印
            print("🔬 各深度神经层【第一主导宗门】分布扫描：")
            for idx, top_cid, count in culprit_per_layer:
                c_name = CLUSTER_NAMES[top_cid]
                warning = " 🔥 [极度可疑]" if top_cid == 16 else ""
                print(
                    f"   - Layer {idx:02d} : #{top_cid:02d} [{c_name:<16}] (活跃 {count:3d} 拍){warning}"
                )

            print("─" * 70)
            print("💡 审判处置建议：")
            print("   👉 输入 /cage 16      ➔ 把 16 宗门关禁闭（临时完全静音）")
            print("   👉 输入 /snipe 21 16  ➔ 狙杀第 21 层的 16 宗门（定点切除）")
            print("═" * 70 + "\n")
            continue

        # ═══════════════════════════════════════════════════════════════
        # 🔒 2. 关禁闭：/cage <宗门号> (临时失效，随时可恢复)
        # ═══════════════════════════════════════════════════════════════
        if user_input.startswith("/cage"):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法: /cage <宗门号> (例如: /cage 16)")
                continue
            cid = int(parts[1])
            if cid not in caged_storage:
                caged_storage[cid] = []
                for layer in model.model.layers:
                    # 备份当前权重，然后就地清零
                    caged_storage[cid].append(
                        layer.mlp.lora_B_cpu[cid].clone())
                    layer.mlp.lora_B_cpu[cid].zero_()
                print(
                    f"🔒 [已关禁闭] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 已被全面封印！立即失效！\n"
                )
            else:
                print(f"⚠️ 宗门 #{cid:02d} 已经在禁闭室了！")
            continue

        # ═══════════════════════════════════════════════════════════════
        # 🔓 3. 刑满释放：/free <宗门号> (恢复原状)
        # ═══════════════════════════════════════════════════════════════
        if user_input.startswith("/free"):
            parts = user_input.split()
            cid = int(parts[1])
            if cid in caged_storage:
                for idx, layer in enumerate(model.model.layers):
                    layer.mlp.lora_B_cpu[cid].copy_(caged_storage[cid][idx])
                del caged_storage[cid]
                print(
                    f"🔓 [刑满释放] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 已恢复全额算力！\n"
                )
            else:
                print(f"⚠️ 宗门 #{cid:02d} 并没有被关押！")
            continue

        # ═══════════════════════════════════════════════════════════════
        # 🎯 4. 单层精准狙杀：/snipe <层数> <宗门号>
        # ═══════════════════════════════════════════════════════════════
        if user_input.startswith("/snipe"):
            parts = user_input.split()
            l_idx = int(parts[1])
            cid = int(parts[2])
            model.model.layers[l_idx].mlp.lora_B_cpu[cid].zero_()
            print(f"🎯 [狙击完毕] 第 {l_idx} 层的 #{cid} 宗门已被单点物理击毙！\n")
            continue

        # ═══════════════════════════════════════════════════════════════
        # 🔌【核心科技：赛博义体在线热插拔通道】
        # 用法示例：/plug cartridge_gongfang.pt
        # ═══════════════════════════════════════════════════════════════
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

            # 🌟 核心内存手术：直接原地 copy_ 改写 CPU 锁页内存中的宗门！
            for i, layer in enumerate(model.model.layers):
                layer_data = cart["layers"][i]
                layer.mlp.lora_A_cpu[target_slot].copy_(layer_data["lora_A"])
                layer.mlp.lora_B_cpu[target_slot].copy_(layer_data["lora_B"])
                # 动态刷新该层路由器的特征向量
                layer.mlp.router_cluster.weight.data[target_slot].copy_(
                    layer_data["router_vec"].to("cuda:0"))

            elapsed_ms = (time.perf_counter() - t_plug) * 1000
            print(f"\n⚡ [热插拔成功] 技能卡带《{cart_name}》已就地植入插槽 #{target_slot:02d}！")
            print(f"   ⏱️  注入耗时: {elapsed_ms:.2f} ms | GPU 零抖动 | 立即生效！\n")
            continue

        for layer in model.model.layers:
            layer.mlp.reset_stats()

        messages.append({"role": "user", "content": user_input})
        prompt_text = tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda:0")

        streamer = TextIteratorStreamer(tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)

        generation_kwargs = dict(**inputs,
                                 streamer=streamer,
                                 max_new_tokens=2048,
                                 do_sample=True,
                                 temperature=0.7,
                                 top_p=0.9,
                                 repetition_penalty=1.15,
                                 eos_token_id=[tokenizer.eos_token_id, 151645])

        print("\n🤖 Assistant: ", end="", flush=True)

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        t0 = time.perf_counter()
        thread.start()

        accumulated_text = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            accumulated_text += chunk
        thread.join()

        elapsed_sec = time.perf_counter() - t0
        gen_tokens = len(
            tokenizer.encode(accumulated_text, add_special_tokens=False))
        speed = gen_tokens / elapsed_sec if elapsed_sec > 0 else 0
        print(
            f"\n\n⚡ 速度: {speed:.1f} tokens/s (共 {gen_tokens} 字, 耗时 {elapsed_sec*1000:.0f} ms)"
        )

        show_myriad_dashboard(model)
        messages.append({"role": "assistant", "content": accumulated_text})


if __name__ == "__main__":
    main()

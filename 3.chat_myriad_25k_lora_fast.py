import os
import sys
import time
from threading import Thread
from collections import Counter
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 🌟 20 宗门定义（含 4 大预留插槽）
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
    # 16~19: 自定义热插拔特区
    "Custom_Rules",  # #16: 业务/工坊硬规则插槽 (预备替换)
    "Custom_Knowledge",  # #17: 私有知识库插槽
    "Custom_Persona",  # #18: 专属人设/角色插槽
    "Custom_Logic"  # #19: 专属推理/任务插槽
]


# ═══════════════════════════════════════════════════════════════
# 🌟 极速优化版推理包装层：全显存常驻 + 零阻断统计 + 电磁屏蔽偏置网
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
                 reside_on_gpu=True):  # 🌟 默认开启 5090 D 显存全常驻模式 (仅吃 1.6GB)
        super().__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.experts_per_cluster = experts_per_cluster
        self.micro_rank = micro_rank
        self.macro_rank = macro_rank
        self.reside_on_gpu = reside_on_gpu

        # [L0 底座]: 唯一底盘 MLP，彻底免去 deepcopy
        self.base_mlp = original_mlp.to(device)

        # [L1 理科大核]: 高阶 LoRA (macro_rank=64)
        self.sci_lora_A = nn.Linear(hidden_dim,
                                    macro_rank,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)
        self.sci_lora_B = nn.Linear(macro_rank,
                                    hidden_dim,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)

        # 路由器
        self.router_big = nn.Linear(hidden_dim,
                                    2,
                                    bias=False,
                                    device=device,
                                    dtype=dtype)
        self.router_cluster = nn.Linear(hidden_dim,
                                        num_clusters,
                                        bias=False,
                                        device=device,
                                        dtype=dtype)

        # 专家权重池引用
        self.lora_A_gpu = None
        self.lora_B_gpu = None
        self.lora_A_cpu = None
        self.lora_B_cpu = None

        # 针对 CPU 模式保留的传输流和静态复用 Buffer
        if not self.reside_on_gpu:
            self.transfer_stream = torch.cuda.Stream(device=device)
            self.buf_A1 = torch.empty(
                (experts_per_cluster, micro_rank, hidden_dim),
                device=device,
                dtype=dtype)
            self.buf_B1 = torch.empty(
                (experts_per_cluster, hidden_dim, micro_rank),
                device=device,
                dtype=dtype)
            self.buf_A2 = torch.empty(
                (experts_per_cluster, micro_rank, hidden_dim),
                device=device,
                dtype=dtype)
            self.buf_B2 = torch.empty(
                (experts_per_cluster, hidden_dim, micro_rank),
                device=device,
                dtype=dtype)

        # 🌟 GPU 原地计数器 (彻底消灭循环内 .item() 强行同步)
        self.register_buffer(
            "stat_arts_weight",
            torch.zeros(1, device=device, dtype=torch.float32))
        self.register_buffer(
            "stat_sci_weight",
            torch.zeros(1, device=device, dtype=torch.float32))
        self.register_buffer(
            "stat_cluster_counts",
            torch.zeros(num_clusters, device=device, dtype=torch.int32))

        # 🌟 禁闭电磁屏蔽偏置网 (平时为 0，关禁闭/狙击时被打入 -1e4 冷宫)
        self.register_buffer(
            "cluster_bias",
            torch.zeros(num_clusters, device=device, dtype=dtype))

    def reset_stats(self):
        self.stat_arts_weight.zero_()
        self.stat_sci_weight.zero_()
        self.stat_cluster_counts.zero_()

    def forward(self, x):
        b, s, d = x.shape
        current_token = x[:, -1:, :]

        # 1. 双大核文理动态调度 (纯 GPU 原地无锁累加)
        logits_big = self.router_big(current_token)
        w_big = torch.softmax(logits_big, dim=-1)
        self.stat_arts_weight += w_big[0, 0, 0]
        self.stat_sci_weight += w_big[0, 0, 1]

        # 基座前向 + 理科增量前向
        base_out = self.base_mlp(x)
        sci_delta = 0.1 * self.sci_lora_B(self.sci_lora_A(x))
        big_out = base_out + (w_big[..., 1:2] * sci_delta)

        # 2. 20 宗门动态路由 (叠加 cluster_bias 屏蔽网)
        logits_cluster = self.router_cluster(current_token) + self.cluster_bias
        w_cluster = torch.softmax(logits_cluster, dim=-1)
        top2_scores, top2_clusters = torch.topk(w_cluster, k=2, dim=-1)

        c1 = top2_clusters[0, 0, 0]
        c2 = top2_clusters[0, 0, 1]

        # GPU 原地记录热度 (零 CPU 中断)
        self.stat_cluster_counts[c1] += 1
        self.stat_cluster_counts[c2] += 1

        # 3. 专家切片获取 (全显存模式下为零开销 Tensor 切片指针)
        if self.reside_on_gpu:
            A1, B1 = self.lora_A_gpu[c1], self.lora_B_gpu[c1]
            A2, B2 = self.lora_A_gpu[c2], self.lora_B_gpu[c2]
        else:
            c1_idx, c2_idx = c1.item(), c2.item()
            with torch.cuda.stream(self.transfer_stream):
                self.buf_A1.copy_(self.lora_A_cpu[c1_idx], non_blocking=True)
                self.buf_B1.copy_(self.lora_B_cpu[c1_idx], non_blocking=True)
                self.buf_A2.copy_(self.lora_A_cpu[c2_idx], non_blocking=True)
                self.buf_B2.copy_(self.lora_B_cpu[c2_idx], non_blocking=True)
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            A1, B1 = self.buf_A1, self.buf_B1
            A2, B2 = self.buf_A2, self.buf_B2

        # 4. 【极速执行区】：区分 Prefill 与 Decode 阶段
        M_len = b * s
        scale = 1.0 / self.experts_per_cluster

        if M_len > 1:
            # Prefill 阶段 (长 Prompt): 走并行 einsum
            h1 = torch.einsum('bsd,erd->bser', x, A1)
            out1 = torch.sum(torch.einsum('bser,edr->bsed', h1, B1),
                             dim=2) * scale

            h2 = torch.einsum('bsd,erd->bser', x, A2)
            out2 = torch.sum(torch.einsum('bser,edr->bsed', h2, B2),
                             dim=2) * scale
        else:
            # 🌟 Decode 阶段 (M=1 极致优化):
            # 消除 Padding 和 atomic_add 冲突，直接利用原生 Batch-GEMV 向量化执行
            x_vec = x.view(d, 1)

            # 专家 1 前向
            h1 = torch.matmul(A1, x_vec)  # [45, r, 1]
            o1 = torch.matmul(B1, h1)  # [45, d, 1]
            out1 = (torch.sum(o1, dim=0) * scale).view(b, s, d)

            # 专家 2 前向
            h2 = torch.matmul(A2, x_vec)
            o2 = torch.matmul(B2, h2)
            out2 = (torch.sum(o2, dim=0) * scale).view(b, s, d)

        micro_out = top2_scores[..., 0:1] * out1 + top2_scores[..., 1:2] * out2
        return big_out + 0.025 * micro_out


def show_myriad_dashboard(model):
    # 汇总各层 GPU 统计量 (仅在生成结束调用一次，完全不影响生成速度)
    total_arts = sum(layer.mlp.stat_arts_weight.item()
                     for layer in model.model.layers)
    total_sci = sum(layer.mlp.stat_sci_weight.item()
                    for layer in model.model.layers)
    all_big = total_arts + total_sci
    arts_pct = (total_arts / all_big * 100) if all_big > 0 else 50
    sci_pct = (total_sci / all_big * 100) if all_big > 0 else 50

    total_counts = torch.zeros(20, dtype=torch.int32)
    for layer in model.model.layers:
        total_counts += layer.mlp.stat_cluster_counts.cpu()

    print("\n" + "═" * 70)
    print("🌌【Myriad-MoE: 25,200 微专家全息透视 (极速流水线版)】:")
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

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 dtype=dtype,
                                                 device_map="cuda:0")
    hidden_dim = model.config.hidden_size

    # 🌟 5090 D 显存充足，开启常驻模式
    USE_GPU_RESIDENT = True

    for layer in model.model.layers:
        layer.mlp = MyriadInferenceWrapper(layer.mlp,
                                           hidden_dim=hidden_dim,
                                           num_clusters=20,
                                           experts_per_cluster=45,
                                           micro_rank=16,
                                           macro_rank=64,
                                           device="cuda:0",
                                           dtype=dtype,
                                           reside_on_gpu=USE_GPU_RESIDENT)

    print(f"[*] 正在挂载 25,200 个微专家...")
    saved = torch.load(weights_path, map_location="cpu")

    for i, layer in enumerate(model.model.layers):
        layer.mlp.sci_lora_A.load_state_dict(saved[f"layer_{i}_sci_lora_A"])
        layer.mlp.sci_lora_B.load_state_dict(saved[f"layer_{i}_sci_lora_B"])
        layer.mlp.router_big.load_state_dict(saved[f"layer_{i}_router_big"])
        layer.mlp.router_cluster.load_state_dict(
            saved[f"layer_{i}_router_cluster"])

        if USE_GPU_RESIDENT:
            layer.mlp.lora_A_gpu = saved[f"layer_{i}_lora_A"].to("cuda:0",
                                                                 dtype=dtype)
            layer.mlp.lora_B_gpu = saved[f"layer_{i}_lora_B"].to("cuda:0",
                                                                 dtype=dtype)
        else:
            layer.mlp.lora_A_cpu = saved[f"layer_{i}_lora_A"].pin_memory()
            layer.mlp.lora_B_cpu = saved[f"layer_{i}_lora_B"].pin_memory()

    print("\n✅ 两万五千微专家已全员就位 (GPU 纯显存零开销调度已就绪)！")
    print("👉 提示：输入 'clear' 重置记忆，输入 'exit' 退出\n")

    messages = [{
        "role":
        "system",
        "content":
        "You are a master of all domains with 25,200 modular micro-experts."
    }]
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

        # 🚨 1. 抓内鬼雷达：/catch (极速 GPU 显存版)
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
                    print(
                        f"   - Layer {idx:02d} : #{top_cid:02d} [{c_name:<16}] (活跃 {cnt:3d} 拍){warning}"
                    )
                else:
                    print(f"   - Layer {idx:02d} : 暂无前向激活数据 (请先发一句话跑一次推理)")

            print("─" * 70)
            print("💡 审判处置建议：")
            print("   👉 输入 /cage 16      ➔ 把 16 宗门关禁闭（全层权重置零 + 路由打入冷宫）")
            print("   👉 输入 /snipe 21 16  ➔ 狙杀第 21 层的 16 宗门（定点单层切除 + 路由屏蔽）")
            print("═" * 70 + "\n")
            continue

        # 🔒 2. 关禁闭：/cage <宗门号> (声带物理切除 + 路由屏蔽)
        if user_input.startswith("/cage"):
            parts = user_input.split()
            if len(parts) < 2:
                print("⚠️ 用法: /cage <宗门号> (例如: /cage 8)")
                continue
            cid = int(parts[1])
            if cid not in caged_storage:
                caged_storage[cid] = []
                for layer in model.model.layers:
                    target_b = layer.mlp.lora_B_gpu if USE_GPU_RESIDENT else layer.mlp.lora_B_cpu
                    caged_storage[cid].append(target_b[cid].clone())
                    target_b[cid].zero_()
                    # 🌟 激活电磁屏蔽：将该宗门 logits 压制到 -1e4
                    layer.mlp.cluster_bias[cid] = -1e4
                print(
                    f"🔒 [已关禁闭] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 权重已清零 + 路由已被打入冷宫！立即生效！\n"
                )
            else:
                print(f"⚠️ 宗门 #{cid:02d} 已经在禁闭室了！")
            continue

        # 🔓 3. 刑满释放：/free <宗门号>
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
                    # 🌟 解除电磁屏蔽
                    layer.mlp.cluster_bias[cid] = 0.0
                del caged_storage[cid]
                print(
                    f"🔓 [刑满释放] 宗门 #{cid:02d} [{CLUSTER_NAMES[cid]}] 已恢复全额算力与路由权限！\n"
                )
            else:
                print(f"⚠️ 宗门 #{cid:02d} 并没有被关押！")
            continue

        # 🎯 4. 单层精准狙杀：/snipe <层数> <宗门号>
        if user_input.startswith("/snipe"):
            parts = user_input.split()
            if len(parts) < 3:
                print("⚠️ 用法格式: /snipe <层数> <宗门号> (例如: /snipe 21 16)")
                continue
            l_idx = int(parts[1])
            cid = int(parts[2])
            target_b = model.model.layers[
                l_idx].mlp.lora_B_gpu if USE_GPU_RESIDENT else model.model.layers[
                    l_idx].mlp.lora_B_cpu
            target_b[cid].zero_()
            # 🌟 单层路由也打入冷宫
            model.model.layers[l_idx].mlp.cluster_bias[cid] = -1e4
            print(f"🎯 [狙击完毕] 第 {l_idx} 层的 #{cid} 宗门权重已被原地清零，且本层路由已封杀！\n")
            continue

        # 🔌 5. 赛博义体在线热插拔：/plug <卡带文件名.pt> [插槽号]
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
                    layer.mlp.lora_A_gpu[target_slot].copy_(
                        layer_data["lora_A"].to("cuda:0"))
                    layer.mlp.lora_B_gpu[target_slot].copy_(
                        layer_data["lora_B"].to("cuda:0"))
                else:
                    layer.mlp.lora_A_cpu[target_slot].copy_(
                        layer_data["lora_A"])
                    layer.mlp.lora_B_cpu[target_slot].copy_(
                        layer_data["lora_B"])
                layer.mlp.router_cluster.weight.data[target_slot].copy_(
                    layer_data["router_vec"].to("cuda:0"))
                # 确保插槽未被屏蔽
                layer.mlp.cluster_bias[target_slot] = 0.0

            elapsed_ms = (time.perf_counter() - t_plug) * 1000
            print(f"\n⚡ [热插拔成功] 技能卡带《{cart_name}》已就地植入插槽 #{target_slot:02d}！")
            print(f"   ⏱️  注入耗时: {elapsed_ms:.2f} ms | 零抖动 | 立即生效！\n")
            continue

        # 统计重置
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

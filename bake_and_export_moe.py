#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
🌌【Myriad-MoE ➔ 官方标准 Qwen3-MoE 模型导出器】
═══════════════════════════════════════════════════════════════════════════════

将 20 宗门微专家 + 文理双大核，离线物化为标准 Qwen3-MoE 架构：
  · 架构标准 : Qwen3MoeForCausalLM（transformers / vLLM / sglang 原生支持）
  · 零自定义依赖，不需要 trust_remote_code

═══════════════════════════════════════════════════════════════════════════════
① 为什么目标架构是 Qwen3Moe 而不是 Qwen2Moe（这是本脚本最关键的一处）
═══════════════════════════════════════════════════════════════════════════════
底座是 Qwen3-0.6B，它的注意力里有 **per-head RMSNorm**：

    self_attn.q_norm : Qwen3RMSNorm(head_dim)     ← Qwen3 有
    self_attn.k_norm : Qwen3RMSNorm(head_dim)

而 Qwen2Moe 的注意力**没有** q_norm / k_norm。若把 Qwen3 的权重搬进
Qwen2Moe，这两层归一化会被静默丢弃——拷权重时不报错（形状本来就对），
但模型行为已经不是那个模型了，输出直接退化成 "QuestionQuestion…" 之类。

Qwen3Moe 与 Qwen3 底座结构完全对齐（同样有 q_norm/k_norm、同样走
Qwen3MoeTopKRouter），所以它是唯一正确的落盘目标。

⚠ 另注：Qwen2MoeSparseMoeBlock 会**无条件**给 shared expert 加一个 sigmoid
门，而 Qwen3Moe 干脆没有 shared expert。本脚本不依赖 shared expert，
两者都不需要碰。

═══════════════════════════════════════════════════════════════════════════════
② 关于 norm_topk_prob —— 必须为 True，否则导出的模型是坏的
═══════════════════════════════════════════════════════════════════════════════
官方 MoE 块的输出是

    out = Σ_k w_k · expert_k(x)

top-k 是从 20 个专家的 softmax 里挑的，Σ_k w_k 远小于 1。本脚本把底座 MLP
复制进全部 routed expert（这是闭式吸收 Δ 的前提：吸收只能改 down_proj，
gate/up 必须能逐元素拷贝），于是

    out = Σ_k w_k · (base_mlp(x) + Δ_sci + Δ_k)
        = (Σ_k w_k) · [base_mlp(x) + Δ_sci]     ← 整条稠密底座被乘了 Σ w_k
        + Σ_k w_k · Δ_k

而训练态里 base_mlp 与 Δ_sci 的系数恒为 1.0。实测本项目 checkpoint
（top-2 / 20）：L0 仅 0.185、L5 0.231、L13 0.438、L27 0.757，平均 0.40
—— 导出模型会丢掉 60% 的底座。

置 norm_topk_prob=True 后路由器会做 w_k /= Σw_k，Σ_k w_k ≡ 1，底座权重回到
1.0，与训练态一致。代价是专家内部的相对权重被重归一化（训练态用原始
softmax 概率），这是标准 MoE 蒸馏固有的取舍，远优于削掉主干。

═══════════════════════════════════════════════════════════════════════════════
③ 声明的近似（会写进 manifest）
═══════════════════════════════════════════════════════════════════════════════
  1. w_sci(x) 是逐 token 标量 → 冻结为校准集上的实测均值 E[w_sci]。
     实测该门控在单层内可在 0.03~0.99 摆动，这是一次真实的降级。
  2. 稀疏 top-k 保留官方 MoE 路由，未稠密化。
  3. 体积必然很大：官方 MoE 无权重共享，底座被复制 N 份。

用法：
    python bake_and_export_moe.py --output-dir ./myriad_qwen3_moe
    python bake_and_export_moe.py --calib-file ./corpus.txt --output-dir ./out
    python bake_and_export_moe.py --cage-clusters 12,13 --output-dir ./out
    python bake_and_export_moe.py --plug 16=rules.pt --output-dir ./out
"""

import argparse
import os
import time

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM,
)

from bake_common import (
    CLUSTER_NAMES, RidgeSolver, build_calibration_texts, calibrate,
    calibration_health, compute_activation, measure_w_sci,
    validate_cluster_ids, write_manifest,
)

# MoE 侧为控制耗时不做逐层 λ 择优，取单一经验值（解与专家无关，可整层复用）
RIDGE_LAMBDA = 1e-4

# --offline 时使用的经验门控（无真实语料可用时的兜底）
OFFLINE_FALLBACK_WSCI = 0.45


def build_parser():
    p = argparse.ArgumentParser(
        description="Myriad-MoE 导出为官方标准 Qwen3-MoE 模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="基础底座 ID 或路径")
    p.add_argument("--weights", default="myriad_moe_hierarchical_weights.pt", help="特训权重路径")
    p.add_argument("--output-dir", default="./myriad_official_qwen3_moe", help="导出目录")
    p.add_argument("--device", default="cuda:0", help="计算设备")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"], help="保存精度")

    p.add_argument("--num-experts-per-tok", type=int, default=2, help="Top-K 激活专家数")
    p.add_argument("--sci-weight", type=float, default=None,
                   help="覆盖实测门控均值（对所有层统一）")
    p.add_argument("--micro-scale", type=float, default=0.025, help="微专家缩放因子")
    p.add_argument("--cage-clusters", type=str, default=None,
                   help="封印的宗门(逗号分隔)。封印=从导出模型里彻底移除该专家")
    p.add_argument("--plug", action="append", metavar="[slot=]path.pt", help="卡带挂载，可重复")

    p.add_argument("--calib-tokens", type=int, default=6000, help="内置校准语料目标 token 量")
    p.add_argument("--calib-file", type=str, default=None, help="真实校准语料（每行一段）")
    p.add_argument("--offline", action="store_true", help="跳过真实前向校准（退化模式）")
    p.add_argument("--verify-tokens", type=int, default=512,
                   help="导出后用于数值验真的 token 数（0 关闭）")
    return p


def parse_plug(specs):
    """'16=rules.pt' 与 'rules.pt' 都要能吃。"""
    out = {}
    for spec in specs or []:
        slot_str, sep, path = spec.partition("=")
        if not sep:
            slot_str, path = "16", slot_str
        slot = int(slot_str)
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到卡带: {path}")
        out[slot] = path
    return out


def main():
    args = build_parser().parse_args()
    device = args.device
    dtype = getattr(torch, args.dtype)

    print("=" * 75)
    print("🚀【Myriad-MoE ➔ 官方标准 Qwen3-MoE 导出流水线】")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map=device)
    base_model.eval()
    base_cfg = base_model.config
    num_layers = len(base_model.model.layers)
    hidden_dim = base_cfg.hidden_size

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"找不到特训权重 {args.weights}\n"
            f"       训练产出或用 2.train_myriad_25k.py 重新生成。")
    saved = torch.load(args.weights, map_location="cpu")

    # 形状全部从权重推导，不写死 20 / 45 / 28
    num_clusters = saved["layer_0_lora_A"].shape[0]
    experts_per_cluster = saved["layer_0_lora_A"].shape[1]

    plug_paths = parse_plug(args.plug)
    plugged = {}
    for slot, path in plug_paths.items():
        print(f"   🔌 熔铸前挂载卡带: 插槽 #{slot:02d} ← {path}")
        plugged[slot] = torch.load(path, map_location="cpu")

    # ── 封印：训练态是 cluster_bias=-1e4 让该簇不可路由 ────────────────
    # 官方路由器没有 bias 项，无法直接搬 -1e4。改为**从模型里移除该专家**，
    # 这是对「不可路由 + 参数不用」最忠实的还原，且顺带减小体积。
    caged = ({int(x) for x in args.cage_clusters.split(",")}
             if args.cage_clusters else set())
    validate_cluster_ids(caged, num_clusters, "宗门")
    keep = [c for c in range(num_clusters) if c not in caged]
    if not keep:
        raise ValueError("全部宗门都被封印了，无法导出")
    n_experts = len(keep)
    remap = {c: j for j, c in enumerate(keep)}

    print(f"\n[*] 宗门 {num_clusters} 个，封印 {sorted(caged) if caged else '无'}"
          f" → 导出 {n_experts} 个专家")

    # ── 真实前向校准 ──────────────────────────────────────────────────
    features, measured_w_sci, cal_stats = {}, {}, {}
    if not args.offline:
        texts = build_calibration_texts(args.calib_tokens, tokenizer, args.calib_file)
        src = args.calib_file if args.calib_file else "内置合成"
        print(f"\n[*] 真实前向校准：{len(texts)} 篇语料（{src}）")
        features, _, cal_stats = calibrate(
            base_model, tokenizer, texts, device, hidden_dim)
        measured_w_sci = measure_w_sci(features, saved, num_layers, device)
        print("    " + calibration_health(
            cal_stats["tokens_per_layer"], base_cfg.intermediate_size))
        print(f"[✔] 校准完成 {cal_stats['seconds']:.2f}s · "
              f"每层 {cal_stats['tokens_per_layer']} token")
        print(f"    实测门控 L0={measured_w_sci[0]:.4f}  "
              f"L{num_layers // 2}={measured_w_sci[num_layers // 2]:.4f}  "
              f"L{num_layers - 1}={measured_w_sci[num_layers - 1]:.4f}")
    else:
        print("\n[!] --offline：使用经验门控与标准正态伪特征，保真度会明显变差")

    # ── 官方 Qwen3-MoE 配置 ───────────────────────────────────────────
    print(f"\n[*] 构建官方 Qwen3MoeConfig ...")
    moe_dict = base_cfg.to_dict()
    moe_dict.update({
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_experts": n_experts,
        "num_experts_per_tok": min(args.num_experts_per_tok, n_experts),
        # 必须等于底座 intermediate_size：闭式吸收只能改 down_proj，
        # gate/up 必须能与底座逐元素拷贝，只有维数相同才成立。
        "moe_intermediate_size": base_cfg.intermediate_size,
        "decoder_sparse_step": 1,
        # ★ 关键：必须 True，否则底座被乘以 Σ_k w_k（实测平均仅 0.40）
        "norm_topk_prob": True,
    })
    moe_config = Qwen3MoeConfig.from_dict(moe_dict)
    moe_model = Qwen3MoeForCausalLM(moe_config).to(device=device, dtype=dtype)
    moe_model.eval()

    if getattr(moe_config, "head_dim", None) != base_cfg.head_dim:
        raise RuntimeError(
            f"head_dim 不一致：底座 {base_cfg.head_dim} vs MoE "
            f"{getattr(moe_config, 'head_dim', None)}，注意力权重会拷错")

    n_par = sum(p.numel() for p in moe_model.parameters())
    print(f"    参数量 {n_par / 1e9:.3f}B（bf16 约 {n_par * 2 / 2 ** 30:.1f} GiB）")

    print("[*] 对齐全局结构 ...")
    moe_model.model.embed_tokens.weight.data.copy_(
        base_model.model.embed_tokens.weight.data)
    moe_model.model.norm.weight.data.copy_(base_model.model.norm.weight.data)

    # ── 逐层物化 ──────────────────────────────────────────────────────
    # 全程离线计算：专家权重被原地写入，必须关掉梯度
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    print(f"[*] 展开 {num_layers} 层 × {n_experts} 专家 ...")
    used_w_sci, layer_fid = {}, []
    inter = moe_config.moe_intermediate_size

    for i in range(num_layers):
        bl = base_model.model.layers[i]
        ml = moe_model.model.layers[i]

        # 注意力：Qwen3 ↔ Qwen3Moe 结构一致，逐个拷贝（含 q_norm/k_norm）
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            src_p, dst_p = getattr(bl.self_attn, name), getattr(ml.self_attn, name)
            dst_p.weight.data.copy_(src_p.weight.data)
        ml.input_layernorm.weight.data.copy_(bl.input_layernorm.weight.data)
        ml.post_attention_layernorm.weight.data.copy_(
            bl.post_attention_layernorm.weight.data)

        # 路由：Qwen3MoeTopKRouter.weight 形状 [num_experts, hidden]，
        # 与 checkpoint 的 router_cluster 逐元素同构（softmax 在官方侧做）
        router_w = saved[f"layer_{i}_router_cluster"]["weight"]
        router_w = router_w[keep].to(device, dtype=dtype).clone()
        for slot, cart in plugged.items():
            vec = cart["layers"][i].get("router_vec")
            if vec is not None and slot in remap:
                router_w[remap[slot]].copy_(vec.to(device, dtype=dtype))
        ml.mlp.gate.weight.data.copy_(router_w)

        if args.sci_weight is not None:
            eff = args.sci_weight
        elif args.offline:
            eff = OFFLINE_FALLBACK_WSCI
        else:
            eff = measured_w_sci[i]
        used_w_sci[i] = round(eff, 5)

        A_sci = saved[f"layer_{i}_sci_lora_A"]["weight"].to(device, torch.float32)
        B_sci = saved[f"layer_{i}_sci_lora_B"]["weight"].to(device, torch.float32)
        delta_sci = (B_sci @ A_sci) * (0.1 * eff)

        # clone() 不可省：.to() 在 dtype/device 已匹配时返回同一对象，
        # 不 clone 的话插卡带会就地改写 checkpoint，后续层/专家全被污染。
        A_micro = saved[f"layer_{i}_lora_A"].to(device, torch.float32).clone()
        B_micro = saved[f"layer_{i}_lora_B"].to(device, torch.float32).clone()
        for slot, cart in plugged.items():
            if i >= len(cart["layers"]):
                raise ValueError(
                    f"卡带层数不足：{cart.get('name')} 只有 {len(cart['layers'])} 层")
            A_micro[slot] = cart["layers"][i]["lora_A"].to(device, torch.float32)
            B_micro[slot] = cart["layers"][i]["lora_B"].to(device, torch.float32)

        # Z = silu(X·Wgᵀ)*(X·Wuᵀ)，只依赖底座 gate/up 与激活，与专家无关；
        # ZᵀZ 的 Cholesky 也与目标 Δ 无关 → 每层只做一次，N 个专家复用。
        if args.offline:
            X_cal = torch.randn(2048, hidden_dim, device=device) * (hidden_dim ** -0.5)
        else:
            X_cal = features[i]
        Z = compute_activation(bl.mlp.gate_proj.weight.float(),
                               bl.mlp.up_proj.weight.float(), X_cal)
        solver = RidgeSolver(X_cal, Z, RIDGE_LAMBDA)
        base_down = bl.mlp.down_proj.weight

        # Qwen3MoeExperts 是融合布局，没有 experts[cid].gate_proj：
        #   gate_up_proj : [E, 2I, H]  前 I 行 = gate，后 I 行 = up
        #   down_proj    : [E, H, I]  与底座 down_proj.weight 同向
        ex = ml.mlp.experts
        for j in range(n_experts):
            cid = keep[j]
            ex.gate_up_proj.data[j, :inter, :].copy_(bl.mlp.gate_proj.weight.data)
            ex.gate_up_proj.data[j, inter:, :].copy_(bl.mlp.up_proj.weight.data)

            sect = torch.bmm(B_micro[cid], A_micro[cid]).sum(0) * (
                1.0 / experts_per_cluster)
            m_c = delta_sci + args.micro_scale * sect

            new_down = base_down.float() + solver.delta(m_c)          # [H, I]
            if j == 0:
                layer_fid.append(solver.fidelity(m_c))
            ex.down_proj.data[j].copy_(new_down.to(dtype))

        if (i + 1) % 7 == 0 or (i + 1) == num_layers:
            lo = i - (i % 7)
            seg = layer_fid[lo:i + 1]
            print(f"   - Layer {lo:02d}~{i:02d} 展开完成 "
                  f"(留出集保真度 {sum(seg) / len(seg):.4f})")

    dur = time.perf_counter() - t0
    good = [c for c in layer_fid if c == c]
    avg_fid = sum(good) / len(good) if good else float("nan")
    print(f"[✔] 物化完成 {dur:.1f}s | 全局留出集保真度 {avg_fid:.4f} | "
          f"最低层 {min(good) if good else float('nan'):.4f}")

    # ── 数值验真：导出模型的 MoE 块输出 vs 独立手算参考 ────────────────
    if args.verify_tokens > 0 and not args.offline:
        print(f"\n[*] 数值验真（{args.verify_tokens} token，抽 4 层）...")
        errs = []
        for i in (0, num_layers // 3, 2 * num_layers // 3, num_layers - 1):
            Xv = features[i][:args.verify_tokens].float()
            # 模块权重是 bf16，喂进去的激活必须同 dtype，否则 linear 报 dtype 不符
            actual = moe_model.model.layers[i].mlp(
                Xv.to(dtype).unsqueeze(0)).squeeze(0).float()

            bl = base_model.model.layers[i]
            Zv = compute_activation(bl.mlp.gate_proj.weight.float(),
                                    bl.mlp.up_proj.weight.float(), Xv)
            logits = Xv @ moe_model.model.layers[i].mlp.gate.weight.data.float().T
            probs = torch.softmax(logits, dim=-1)
            w, idx = torch.topk(probs, moe_config.num_experts_per_tok, dim=-1)
            w = w / w.sum(-1, keepdim=True)          # norm_topk_prob=True
            ref = torch.zeros_like(actual)
            for k in range(moe_config.num_experts_per_tok):
                for e in idx[:, k].unique().tolist():
                    m = (idx[:, k] == e)
                    ref[m] += w[m, k : k + 1] * (
                        Zv[m] @ moe_model.model.layers[i].mlp.experts.down_proj.data[e].float().T)
            cos = torch.nn.functional.cosine_similarity(
                actual.flatten(), ref.flatten(), dim=0).item()
            errs.append((i, cos))
            print(f"    L{i:02d}: 实际 vs 手算 参考 cosine = {cos:.6f}")
        if min(c for _, c in errs) < 0.999:
            raise RuntimeError(f"数值验真失败：{errs}，导出结果不可信")
        print("[✔] 数值验真通过：MoE 块输出与独立手算一致")

    # ── 落盘 ──────────────────────────────────────────────────────────
    print(f"\n[*] 导出至 {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    moe_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "baked_from_base": args.base_model,
        "weights_file": args.weights,
        "format": "Qwen3MoeForCausalLM",
        "architecture_note":
            "目标架构必须是 Qwen3Moe：底座 Qwen3 注意力含 per-head RMSNorm "
            "(q_norm/k_norm)，Qwen2Moe 无此结构，搬权重时会静默丢失归一化。",
        "mode": "offline_synthetic" if args.offline else (
            "real_corpus" if args.calib_file else "real_forward_synthetic_corpus"),
        "num_experts": n_experts,
        "num_clusters_trained": num_clusters,
        "experts_per_cluster": experts_per_cluster,
        "top_k": moe_config.num_experts_per_tok,
        "norm_topk_prob": True,
        "params_billion": round(n_par / 1e9, 3),
        "calibration": {
            "tokens_per_layer": cal_stats.get("tokens_per_layer", 0),
            "num_docs": cal_stats.get("num_docs", 0),
            "seconds": cal_stats.get("seconds", 0),
            "ridge_lambda": RIDGE_LAMBDA,
        },
        "per_layer_sci_weight": used_w_sci,
        "per_layer_val_fidelity": [round(c, 5) if c == c else None for c in layer_fid],
        "mean_val_fidelity": round(avg_fid, 5),
        "cluster_order": [CLUSTER_NAMES[c] for c in keep],
        "caged_clusters": [CLUSTER_NAMES[c] for c in sorted(caged)],
        "plugged_cartridges": {str(s): os.path.basename(p) for s, p in plug_paths.items()},
        "approximations": [
            "w_sci(x) 逐 token 动态门控 → 冻结为校准集实测均值",
            "稀疏 top-k 保留官方 MoE 路由，未稠密化",
            "norm_topk_prob=True 使 Σ_k w_k ≡ 1，底座权重与训练态一致",
            "封印宗门以「移除专家」实现（官方路由器无 bias，无法直接搬 -1e4）",
        ],
        "size_note": "官方 MoE 无权重共享，底座 MLP 被复制 num_experts 份；"
                     "moe_intermediate_size 必须等于底座 intermediate_size，"
                     "否则 gate/up 无法逐元素拷贝、闭式吸收不成立",
    }
    path = write_manifest(args.output_dir, "myriad_moe_manifest.json", manifest)

    print("=" * 75)
    print("🎉【官方标准 Qwen3-MoE 模型已交付】")
    print(f"📦 导出路径: {args.output_dir}")
    print(f"📄 配方档案: {path}")
    print(f"🏛️ {n_experts} 专家, Top-{moe_config.num_experts_per_tok}, "
          f"norm_topk_prob=True, {n_par / 1e9:.2f}B")
    print("=" * 75)

    del base_model, moe_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ── 落盘后回读自测 ────────────────────────────────────────────────
    print("\n[*] 回读落盘权重自测 ...")
    test_model = AutoModelForCausalLM.from_pretrained(
        args.output_dir, dtype=dtype, device_map=device).eval()
    for prompt in ("白日依山尽", "中国的首都是", "1 + 1 ="):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            o = test_model.generate(**enc, max_new_tokens=24, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
        txt = tokenizer.decode(o[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🤖 「{prompt}」→ {txt.strip()}")


if __name__ == "__main__":
    main()

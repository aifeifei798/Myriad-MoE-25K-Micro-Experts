#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
🛠️【Myriad-MoE 纯血原生 Dense 模型烘焙固化引擎】
═══════════════════════════════════════════════════════════════════════════════

将 25,200 微专家 + 文理双大核的残差投影，闭式吸收进官方原生 Dense 模型的
down_proj，导出成零自定义依赖的标准 Dense 权重。

导出结果：
  · 100% 官方标准架构 (Qwen2ForCausalLM / Qwen3ForCausalLM)
  · 零自定义代码依赖 (无需 trust_remote_code=True)

═══════════════════════════════════════════════════════════════════════════════
与训练态的结构对应关系（务必读，否则会误判保真度）
═══════════════════════════════════════════════════════════════════════════════
训练/推理时的每层残差是：

    out = base_mlp(x)                          ← 稠密底座，权重恒为 1.0
        + w_sci(x) · 0.1 · Δ_sci · x            ← 文理门控，**逐 token 变化**
        + 0.025 · Σ_k w_k · Δ_{c(k)} · x        ← 稀疏 top-k 微专家

Dense 权重无法表达 top-k 稀疏，也无法表达逐 token 门控，因此烘焙做了两处
**明确声明的近似**（会写进 manifest）：

  1. w_sci(x) 是逐 token 标量 → 用校准集上的实测均值 E[w_sci] 代替。
     这是把「动态路由」冻结成「静态权重」。实测该门控在单层内可在
     0.03~0.99 间摆动，所以这是一次真实的降级，而不是等价变换。
     --sci-weight 可手动覆盖。

  2. 稀疏 top-k → 参与烘焙的所有宗门等权平均（1/|active|）。
     训练态只激活 top_k 个；这里等效为稠密集成。

除以上两点，底座 MLP 本身是被精确保留的（Δ 只加在 down_proj 上）。

用法：
    # 默认：真实前向校准 + 逐层实测门控
    python bake_and_merge_dense.py --output-dir ./qwen_dense

    # 用真实语料校准（强烈建议，合成语料只是兜底）
    python bake_and_merge_dense.py --calib-file ./my_corpus.txt --output-dir ./qwen_dense

    # 强化理科 / 人文
    python bake_and_merge_dense.py --preset code --output-dir ./qwen_code_dense

    # 预插卡带（两种写法都支持）
    python bake_and_merge_dense.py --plug 16=rules.pt --output-dir ./out
    python bake_and_merge_dense.py --plug rules.pt    --output-dir ./out
"""

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from bake_common import (
    CLUSTER_NAMES, DEFAULT_LAMBDAS, absorb_into, build_calibration_texts,
    calibrate, calibration_health, fit_delta, measure_w_sci,
    validate_cluster_ids, write_manifest,
)

# 预设只调「整体倾向」，实测门控仍按层测定后再乘以该系数
PRESETS = {
    "code":       {"sci_scale": 1.3, "clusters": [0, 1, 2, 3, 4, 7, 19]},
    "literature": {"sci_scale": 0.4, "clusters": [12, 13, 14, 15, 17]},
    "balanced":   {"sci_scale": 1.0, "clusters": None},   # None = 全部
    "custom":     {"sci_scale": 1.0, "clusters": None},
}

# --offline 时使用的经验门控。此前用的是 0.35 这个魔数，
# 留在这里仅因为没有真实语料可用；manifest 会标注 mode=offline_synthetic。
OFFLINE_FALLBACK_WSCI = 0.45


def build_parser():
    p = argparse.ArgumentParser(
        description="Myriad-MoE 一键烘焙固化为纯血 Dense 模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B", help="底座模型 ID 或本地路径")
    p.add_argument("--weights", default="myriad_moe_hierarchical_weights.pt", help="特训权重路径")
    p.add_argument("--output-dir", default="./merged_dense_qwen", help="固化模型输出目录")
    p.add_argument("--device", default="cuda:0", help="计算设备")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"], help="保存精度")

    p.add_argument("--preset", default="balanced", choices=list(PRESETS),
                   help="融合风格")
    p.add_argument("--sci-weight", type=float, default=None,
                   help="覆盖自动测定的门控均值（对所有层统一）")
    p.add_argument("--sci-scale", type=float, default=None,
                   help="在实测门控上再乘的系数（仅对 preset 生效）")
    p.add_argument("--micro-scale", type=float, default=0.0125,
                   help="微专家缩放因子。0.0125 为实测甜点（重复率优于底座）；"
                        "调大会让微专家行为更显著但语言能力退化，详见 README")
    p.add_argument("--active-clusters", type=str, default=None, help="参与烘焙的宗门(逗号分隔)")
    p.add_argument("--cage-clusters", type=str, default=None, help="显式剔除的宗门(逗号分隔)")
    p.add_argument("--plug", action="append", metavar="[slot=]path.pt",
                   help="烘焙前预插卡带，可重复")

    p.add_argument("--calib-tokens", type=int, default=8000,
                   help="内置校准语料的目标 token 量。样本数需显著大于 "
                        "中间维 3072，否则闭式解严重欠定")
    p.add_argument("--calib-file", type=str, default=None,
                   help="真实校准语料（每行一段）。给了就用它，优先于内置合成语料")
    p.add_argument("--val-frac", type=float, default=0.2, help="留出集比例（用于报告真实泛化保真度）")
    p.add_argument("--lambdas", type=str, default=",".join(str(x) for x in DEFAULT_LAMBDAS),
                   help="候选正则化系数，按留出集保真度择优")
    p.add_argument("--offline", action="store_true",
                   help="跳过真实前向校准（退化模式，保真度会明显变差）")
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
    lambdas = tuple(float(x) for x in args.lambdas.split(",") if x.strip())

    print("=" * 75)
    print("🚀【Myriad-MoE 纯血原生 Dense 模型烘焙固化引擎】")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=dtype, device_map=device)
    model.eval()
    hidden_dim = model.config.hidden_size
    num_layers = len(model.model.layers)

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"找不到特训权重 {args.weights}\n"
            f"       训练产出或用 2.train_myriad_25k.py 重新生成。")
    saved = torch.load(args.weights, map_location="cpu")

    # 形状全部从权重推导，不写死 20 / 45
    num_clusters = saved["layer_0_lora_A"].shape[0]
    experts_per_cluster = saved["layer_0_lora_A"].shape[1]

    plug_paths = parse_plug(args.plug)
    plugged = {}
    for slot, path in plug_paths.items():
        print(f"   🔌 熔铸前挂载卡带: 插槽 #{slot:02d} ← {path}")
        plugged[slot] = torch.load(path, map_location="cpu")

    # ── 配方 ──────────────────────────────────────────────────────────
    preset = PRESETS[args.preset]
    sci_scale = args.sci_scale if args.sci_scale is not None else preset["sci_scale"]
    if args.active_clusters:
        active = [int(x) for x in args.active_clusters.split(",")]
    elif preset["clusters"] is not None:
        active = list(preset["clusters"])
    else:
        active = list(range(num_clusters))

    caged = {int(x) for x in args.cage_clusters.split(",")} if args.cage_clusters else set()
    active = [c for c in active if c not in caged]

    # active 与 caged 都要查：封印一个不存在的宗门之前是静默无效的
    validate_cluster_ids(list(active) + list(caged), num_clusters, "宗门")
    if not active:
        raise ValueError("参与烘焙的宗门为空，无法继续")

    # ── 真实前向校准 ──────────────────────────────────────────────────
    features, measured_w_sci, cal_stats = {}, {}, {}
    if not args.offline:
        texts = build_calibration_texts(args.calib_tokens, tokenizer, args.calib_file)
        print(f"\n[*] 真实前向校准：{len(texts)} 篇语料"
              f"{'（来自 ' + args.calib_file + '）' if args.calib_file else '（内置合成）'}")
        features, _, cal_stats = calibrate(
            model, tokenizer, texts, device, hidden_dim)
        measured_w_sci = measure_w_sci(features, saved, num_layers, device)
        print("    " + calibration_health(
            cal_stats["tokens_per_layer"], model.config.intermediate_size))
        print(f"[✔] 校准完成 {cal_stats['seconds']:.2f}s · "
              f"每层 {cal_stats['tokens_per_layer']} token · "
              f"{cal_stats['tokens_per_layer'] / max(1, hidden_dim and 3072):.2f}× 中间维")
        print(f"    实测门控 L0={measured_w_sci[0]:.4f}  "
              f"L{num_layers//2}={measured_w_sci[num_layers//2]:.4f}  "
              f"L{num_layers-1}={measured_w_sci[num_layers-1]:.4f}")
    else:
        print("\n[!] --offline：使用经验门控与标准正态伪特征，保真度会明显变差")

    print(f"\n📋【烘焙配方】")
    print(f"   宗门      : {len(active)} / {num_clusters} 参与")
    print(f"   禁闭剔除  : {sorted(caged) if caged else '无'}")
    print(f"   微专家缩放: {args.micro_scale}")
    print(f"   门控缩放  : 实测 × {sci_scale}" if not args.offline else f"   门控: 经验值 {OFFLINE_FALLBACK_WSCI} × {sci_scale}")
    print("─" * 75)

    # ── 逐层烘焙 ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    layer_fidelity, layer_lambda, used_w_sci = [], [], {}

    # 烘焙是纯离线计算：down_proj 会被原地累加，必须关掉梯度
    torch.set_grad_enabled(False)
    for i in range(num_layers):
        layer = model.model.layers[i]

        # 本层实际使用的门控权重
        if args.sci_weight is not None:
            eff = args.sci_weight
        elif args.offline:
            eff = OFFLINE_FALLBACK_WSCI * sci_scale
        else:
            eff = measured_w_sci[i] * sci_scale
        used_w_sci[i] = round(eff, 5)

        # 理科大核
        A_sci = saved[f"layer_{i}_sci_lora_A"]["weight"].to(device, dtype=torch.float32)
        B_sci = saved[f"layer_{i}_sci_lora_B"]["weight"].to(device, dtype=torch.float32)
        delta_sci = (B_sci @ A_sci) * (0.1 * eff)

        # 微专家
        # clone() 不可省：.to() 在 dtype/device 已匹配时返回同一对象，
        # 不 clone 的话插卡带会就地改写 checkpoint，后续层全被污染。
        A_micro = saved[f"layer_{i}_lora_A"].to(device, dtype=torch.float32).clone()
        B_micro = saved[f"layer_{i}_lora_B"].to(device, dtype=torch.float32).clone()
        for slot, cart in plugged.items():
            if i >= len(cart["layers"]):
                raise ValueError(f"卡带层数不足：{cart.get('name')} 只有 {len(cart['layers'])} 层")
            A_micro[slot] = cart["layers"][i]["lora_A"].to(device, torch.float32)
            B_micro[slot] = cart["layers"][i]["lora_B"].to(device, torch.float32)

        delta_micro = torch.zeros(hidden_dim, hidden_dim, device=device, dtype=torch.float32)
        w_norm = 1.0 / len(active)
        for cid in active:
            sect = torch.bmm(B_micro[cid], A_micro[cid]).sum(0) * (1.0 / experts_per_cluster)
            delta_micro.add_(sect, alpha=w_norm)
        delta_micro.mul_(args.micro_scale)

        M_total = delta_sci + delta_micro

        if args.offline:
            X_cal = torch.randn(2048, hidden_dim, device=device) * (hidden_dim ** -0.5)
        else:
            X_cal = features[i]

        delta_w, val_cos, best_lam, _ = fit_delta(
            layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight,
            layer.mlp.down_proj.weight, M_total, X_cal,
            lambdas=lambdas, val_frac=args.val_frac)
        absorb_into(layer.mlp.down_proj.weight, delta_w)

        layer_fidelity.append(val_cos)
        layer_lambda.append(best_lam)

        if (i + 1) % 7 == 0 or (i + 1) == num_layers:
            lo = i - (i % 7)
            seg = layer_fidelity[lo:i + 1]
            print(f"   - Layer {lo:02d}~{i:02d} 熔炼完成 "
                  f"(留出集保真度 {sum(seg)/len(seg):.4f})")

    dur = time.perf_counter() - t0
    good = [c for c in layer_fidelity if c == c]
    avg_fid = sum(good) / len(good) if good else float("nan")
    print(f"[✔] 烘焙完成 {dur:.1f}s | 全局留出集保真度 {avg_fid:.4f} | "
          f"最低层 {min(good) if good else float('nan'):.4f}")
    if good and min(good) < 0.6:
        print(f"    ⚠ 有层保真度 < 0.60 —— 主因通常是校准语料欠定，"
              f"请加大 --calib-tokens 或改用 --calib-file；"
              f"调低 --micro-scale 通常无效（实测保真度反而随其增大而升高）")

    # ── 落盘 ──────────────────────────────────────────────────────────
    print(f"\n[*] 保存至 {args.output_dir} ...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    manifest = {
        "baked_from_base": args.base_model,
        "weights_file": args.weights,
        "format": "dense",
        "preset": args.preset,
        "mode": "offline_synthetic" if args.offline else (
            "real_corpus" if args.calib_file else "real_forward_synthetic_corpus"),
        "calibration": {
            "tokens_per_layer": cal_stats.get("tokens_per_layer", 0),
            "num_docs": cal_stats.get("num_docs", 0),
            "seconds": cal_stats.get("seconds", 0),
            "val_frac": args.val_frac,
            "lambdas_tried": list(lambdas),
        },
        "per_layer_sci_weight": used_w_sci,
        "per_layer_val_fidelity": [round(c, 5) if c == c else None for c in layer_fidelity],
        "per_layer_best_lambda": layer_lambda,
        "mean_val_fidelity": round(avg_fid, 5),
        "micro_scale": args.micro_scale,
        "experts_per_cluster": experts_per_cluster,
        "active_clusters": [CLUSTER_NAMES[c] for c in active],
        "caged_clusters": sorted(caged),
        "plugged_cartridges": {str(s): os.path.basename(p) for s, p in plug_paths.items()},
        "approximations": [
            "w_sci(x) 逐 token 动态门控 → 冻结为校准集实测均值",
            "稀疏 top-k 微专家 → 参与宗门等权平均 (稠密集成)",
        ],
    }
    path = write_manifest(args.output_dir, "myriad_bake_manifest.json", manifest)

    print("=" * 75)
    print("🎉【纯血原生 Dense 模型已交付】")
    print(f"📦 导出路径: {args.output_dir}")
    print(f"📄 配方档案: {path}")
    print("=" * 75)

    # ── 原生自测 ──────────────────────────────────────────────────────
    for prompt in ("白日依山尽", "中国的首都是"):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        txt = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"🤖 「{prompt}」→ {txt.strip()}")


if __name__ == "__main__":
    main()
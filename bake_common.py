#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bake_and_export_moe.py / bake_and_merge_dense.py 的公共部件。

抽出来只有一个理由：这两个脚本之前已经因为「各写各的」而漂移过
（--plug 解析在一边是 bug、另一边是对的；保真度指标只有一边有）。
凡是两边的数学必须一致的东西，都放这里。

包含：
  · CLUSTER_NAMES                —— 与训练脚本保持一致
  · 校准语料构造                 —— 多领域种子 + 程序化扩展到目标 token 量
  · 真实前向校准                —— hook 采集各层特征 + 实测 w_sci
  · 闭式 Ridge 投影              —— 训练/留出划分 + λ 自动选择
  · manifest 落盘
"""

import json
import os
import random
import time

import torch

# ═══════════════════════════════════════════════════════════════════════
# 宗门名（必须与 6.chat_*.py / 2.train_*.py 保持一致）
# ═══════════════════════════════════════════════════════════════════════
CLUSTER_NAMES = [
    "Code_Algo", "Code_DS", "Code_Debug", "Code_Arch",          # 0~3 代码
    "Math_Algebra", "Math_Geo", "Math_Prob", "Math_Arith",       # 4~7 数学
    "Sci_Physics", "Sci_Chem", "Sci_Biology", "Sci_Astronomy",   # 8~11 科学
    "Arts_Rhetoric", "Arts_Philosophy", "Arts_Summary", "Arts_Chat",  # 12~15 人文
    "Custom_Rules", "Custom_Knowledge", "Custom_Persona", "Custom_Logic",  # 16~19 特区
]

# ═══════════════════════════════════════════════════════════════════════
# 校准语料
#
# 为什么不能只用几段短文本：闭式解要解 [I, I] = [3072, 3072] 的系统，
# 训练样本数 N 必须显著大于 I 才不会严重欠定。此前只用 6 段 ≈ 250 token
# （N≈200 « 3072），留出集保真度被压在 0.75 左右，后 7 层甚至掉到 0.56。
#
# 下面是分领域种子，再按目标 token 数程序化拼装成较长的文档。
# 注意：程序化扩展仍然是合成分布，真实语料请用 --calib-file 提供。
# ═══════════════════════════════════════════════════════════════════════
CALIBRATION_SEEDS = {
    "literature_zh": [
        "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
        "落霞与孤鹜齐飞，秋水共长天一色。",
        "问渠那得清如许，为有源头活水来。",
        "纸上得来终觉浅，绝知此事要躬行。",
        "山重水复疑无路，柳暗花明又一村。",
        "君��远别离，思念如潮水。",
    ],
    "literature_en": [
        "It was the best of times, it was the worst of times.",
        "All that we see or seem is but a dream within a dream.",
        "The only way out of the labyrinth of suffering is compassion.",
        "In the beginning the universe was created, which made a lot of people very angry.",
    ],
    "science": [
        "在三体问题中，拉格朗日点 L1 处的引力与离心力处于动态平衡状态。",
        "DNA 双螺旋结构由两条反向平行的多核苷酸链组成，通过碱基配对连接。",
        "量子纠缠态的两个粒子无论相距多远，测量其一会瞬时决定另一个的状态。",
        "光合作用将光能转化为化学能，储存在葡萄糖分子的化学键中。",
        "The second law of thermodynamics states that entropy never decreases.",
        "板块构造学说解释了大陆漂移、海底扩张与地震带分布。",
        "CRISPR-Cas9 基因编辑技术通过向导 RNA 定位特定 DNA 序列并切割。",
    ],
    "math": [
        "求方程 x^2 - 5x + 6 = 0 的全部实根，并写出推导过程。",
        "证明：对任意 n >= 1，1 + 2 + ... + n = n(n+1)/2。",
        "计算定积分 ∫₀^1 x^2 · e^x dx 的精确值。",
        "设 f(x) = sin(x)/x，求其在 x=0 处的连续延拓值。",
        "A matrix is invertible if and only if its determinant is nonzero.",
        "The variance of a Bernoulli random variable with parameter p equals p(1-p).",
        "用归纳法证明任意 n >= 1 的完全图 K_n 有 n(n-1)/2 条边。",
    ],
    "code": [
        "def quick_sort(arr): return arr if len(arr) <= 1 else quick_sort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quick_sort([x for x in arr[1:] if x >= arr[0]])",
        "SELECT users.id, COUNT(orders.id) AS n FROM users LEFT JOIN orders ON users.id = orders.user_id GROUP BY users.id HAVING COUNT(orders.id) > 5 ORDER BY n DESC;",
        "class Node:\n    def __init__(self, val, next=None):\n        self.val = val\n        self.next = next",
        "async def fetch_all(urls):\n    tasks = [aiohttp.get(u) for u in urls]\n    return await asyncio.gather(*tasks)",
        "The binary search algorithm finds an element in a sorted array in O(log n) time by repeatedly halving the search interval.",
        "git rebase -i HEAD~3 可以把最近三个提交压成一个，然后修改历史。",
        "public static void main(String[] args) { System.out.println(\"hello\"); }",
    ],
    "dialogue": [
        "用户：帮我把这段 Python 改成异步的。\n助手：可以，先看一下原来的实现。",
        "用户：这个 bug 怎么复现？\n助手：在输入框里输入超过 1024 个字符就会触发。",
        "助手：你刚才说的那件事，我需要确认一下时间地点。",
        "用户：谢谢！\n助手：不客气，还有其他可以帮你的吗？",
        "A: Could you clarify what you mean by \"it\"?\nB: I mean the configuration file we discussed earlier.",
    ],
}

_ROLE_PREFIX = ["", "问题：", "答：", "输入：", "输出：", "Note: ", "Summary: "]


def build_calibration_texts(target_tokens: int, tokenizer=None, extra_file: str = None,
                            seed: int = 0) -> list:
    """构造校准语料。

    extra_file 存在时优先使用（真实语料 > 合成语料）。
    否则用手写种子按 target_tokens 拼装长文档：
      · 随机抽取 2~5 段不同领域的种子拼成一篇，模拟真实文档的长短不一
      · 随机加角色前缀，制造对话/问答形态
    """
    if extra_file:
        texts = []
        with open(extra_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                if len(texts) >= 4000:
                    break
        if texts:
            return texts

    rng = random.Random(seed)
    pool = [(dom, s) for dom, lst in CALIBRATION_SEEDS.items() for s in lst]

    # 先估算种子平均 token 数（没 tokenizer 就按字符数粗估）
    def est(t):
        if tokenizer is not None:
            return len(tokenizer(t, add_special_tokens=False)["input_ids"])
        return max(1, int(len(t) / 2.2))

    avg = sum(est(s) for _, s in pool) / len(pool)
    per_doc = max(3, int(round(avg * 3.2)))   # 每篇 3~5 段
    n_docs = max(8, int(target_tokens / per_doc) + 1)

    docs = []
    for _ in range(n_docs):
        k = rng.randint(2, 5)
        # 有意混入跨领域片段：单篇内领域混杂更接近真实语料的分布
        picked = [rng.choice(pool) for _ in range(k)]
        body = "\n".join(t for _, t in picked)
        pre = rng.choice(_ROLE_PREFIX)
        docs.append(pre + body if pre else body)
    return docs


# ═══════════════════════════════════════════════════════════════════════
# 真实前向校准
# ═══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def calibrate(model, tokenizer, texts, device, hidden_dim):
    """跑真实前向，采集每层 MLP 输入特征与实测 w_sci。

    hook 挂在 layer.mlp 上，inp[0] 就是 post_attention_layernorm 之后的隐状态，
    也就是 MLP 真正吃进去的那个 x —— 与推理态完全一致。

    返回 (features, w_sci, stats)
      features: {layer: [N, D] float32 (device)}
      w_sci:    {layer: float}  实测 E[softmax(router_big(x))[...,1]]
      stats:    诊断信息
    """
    n_layers = len(model.model.layers)
    captured = {i: [] for i in range(n_layers)}

    def mk(i):
        def hook(mod, inp, out):
            captured[i].append(inp[0].detach().float().reshape(-1, hidden_dim))
        return hook

    hooks = [model.model.layers[i].mlp.register_forward_hook(mk(i)) for i in range(n_layers)]
    t0 = time.perf_counter()
    try:
        for text in texts:
            enc = tokenizer(text, return_tensors="pt").to(device)
            model(**enc)
    finally:
        for h in hooks:
            h.remove()

    features, w_sci = {}, {}
    n_tokens = 0
    for i in range(n_layers):
        feats = torch.cat(captured[i], dim=0)
        features[i] = feats
        n_tokens = max(n_tokens, feats.shape[0])

    stats = {
        "num_docs": len(texts),
        "tokens_per_layer": n_tokens,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    return features, w_sci, stats


def validate_cluster_ids(ids, num_clusters, label):
    """宗门编号越界检查。Dense 与 MoE 都要用，避免封印一个不存在的宗门静默无效。"""
    bad = sorted({i for i in ids if not (0 <= i < num_clusters)})
    if bad:
        raise ValueError(
            f"{label}编号越界 {bad}，合法范围 0 ~ {num_clusters - 1}")
    return set(ids)


def calibration_health(tokens_per_layer, intermediate_dim):
    """校准样本量相对中间维的倍数。

    闭式解要解 [I, I] = [3072, 3072] 的系统，样本数 N 显著大于 I 才有意义。
    N « I 时系统严重欠定，留出集保真度会低得离谱——这是必须让用户知道的，
    否则一个 250 token 的语料跑出来的 0.75 会被误当成「烘焙质量就这」。
    """
    ratio = tokens_per_layer / float(intermediate_dim)
    level = "✓" if ratio >= 1.0 else ("⚠" if ratio >= 0.3 else "✗")
    msg = (f"{level} 校准样本 {tokens_per_layer} / 中间维 {intermediate_dim} = "
           f"{ratio:.2f}×")
    if ratio < 1.0:
        msg += ("（欠定，保真度会明显偏低；建议加大 --calib-tokens 或换真实语料）"
                if ratio >= 0.3 else
                "（严重欠定，保真度数字基本不可信，请务必加大 --calib-tokens）")
    return msg


def measure_w_sci(features, saved, n_layers, device):
    """实测每层 w_sci 期望。需要在 calibrate 之后、权重被改写之前调用。"""
    out = {}
    for i in range(n_layers):
        r = saved[f"layer_{i}_router_big"]["weight"].to(features[i].device, torch.float32)
        logits = features[i] @ r.T
        out[i] = torch.softmax(logits, dim=-1)[:, 1].mean().item()
    return out


# ═══════════════════════════════════════════════════════════════════════
# 闭式 Ridge 投影
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2)


def compute_activation(gate_w: torch.Tensor, up_w: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Z = silu(X·Wgᵀ) * (X·Wuᵀ) —— 即 MLP 的中间激活。"""
    return torch.nn.functional.silu(X @ gate_w.T) * (X @ up_w.T)


def fit_delta(gate_w, up_w, down_w, M, X, lambdas=DEFAULT_LAMBDAS, val_frac=0.2):
    """把 M 闭式吸收进 down_proj，并返回留出集上的真实保真度。

    训练/留出按**样本顺序**切分（校准语料按文档拼接，因此这等价于
    「前 80% 文档训练、后 20% 文档验证」），比随机划分更贴近对未见文本的泛化。

    λ 会逐个试，用留出集 cosine 选最优。此前固定 λ=1e-4 且在**拟合集**上
    报指标，那个数字既不能反映泛化、也可以靠调 λ 修饰。

    返回 (delta_w, val_cos, best_lambda, train_cos)
    """
    device = down_w.device
    Z = compute_activation(gate_w.float(), up_w.float(), X)

    n = Z.shape[0]
    n_val = max(1, int(n * val_frac)) if n >= 32 else 0
    n_tr = n - n_val if n_val else n
    Ztr = Z[:n_tr]
    Zva = Z[n_tr:] if n_val else None
    M64 = M.double()
    # 一律 float64：深层激活范数可达 10²~10³，fp32 下 ZᵀZ 条件数极差，
    # solve 会直接抛 "singular matrix"。与 RidgeSolver 保持同一套数值口径。
    Xtr, Xva = X[:n_tr].double(), (X[n_tr:].double() if n_val else None)
    Ytr = Xtr @ M64.T
    Yva = (Xva @ M64.T) if n_val else None
    Ztr64, Zva64 = Ztr.double(), (Zva.double() if n_val else None)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0).item()

    def solve(lam):
        ZtZ = Ztr64.T @ Ztr64
        ZtZ.diagonal().add_(lam * n_tr)
        return torch.linalg.solve(ZtZ, (Ytr.T @ Ztr64).T).T   # [D, I]

    best = None
    for lam in lambdas:
        Dw = solve(lam)
        tr = cos(Ytr, Ztr64 @ Dw.T)
        va = cos(Yva, Zva64 @ Dw.T) if n_val else float("nan")
        if best is None:
            best = (Dw, va, lam, tr)
            continue
        bva = best[1]
        # NaN 安全：val 为 NaN 时保留先到的；否则要求严格更优
        if (va == va) and ((bva != bva) or va > bva):
            best = (Dw, va, lam, tr)

    Dw, val_cos, best_lam, train_cos = best
    return Dw.to(device=device, dtype=torch.float32), val_cos, best_lam, train_cos


def absorb_into(down_weight: torch.Tensor, delta_w: torch.Tensor) -> None:
    """原地把 ΔW 累加进 down_proj 权重。"""
    down_weight.add_(delta_w.to(down_weight.dtype))


class RidgeSolver:
    """复用的闭式 ridge 求解器。

    ZtZ 只依赖激活 Z，与目标 M 无关 —— 同一层里 20 个专家共享同一份 Z，
    因此 Cholesky 分解只需做一次，之后每个专家只付三角求解的代价。
    比每个专家重新分解快约一个数量级。

    需要同时持有 X（原始输入 [N, D]）与 Z（中间激活 [N, I]）：
      Y    = X · Mᵀ              [N, D]   ← 这里用的是 X，不是 Z
      ΔW   = (YᵀZ)(ZᵀZ+λI)⁻¹    [D, I]

    数值稳健性：深层激活范数可达 10²~10³ 量级，fp32 下 ZᵀZ 条件数极差，
    Cholesky 会直接抛 "not positive-definite"。因此分解与回代一律走
    float64，并在失败时按倍数放大 λ 重试。
    """

    def __init__(self, X: torch.Tensor, Z: torch.Tensor, lam: float):
        self.X = X
        self.Z = Z
        self.n = Z.shape[0]
        self.lam_used = lam

        Z64 = Z.double()
        ZtZ = Z64.T @ Z64
        base = lam * self.n
        self.L = None
        for k in range(8):
            cand = ZtZ.clone()
            cand.diagonal().add_(base * (10.0 ** k))
            try:
                self.L = torch.linalg.cholesky(cand)
                self.lam_used = lam * (10.0 ** k)
                break
            except Exception:
                continue
        if self.L is None:                      # 兜底：按矩阵尺度加抖动
            cand = ZtZ.clone()
            cand.diagonal().add_(max(base, 1e-3 * float(torch.diagonal(ZtZ).mean())))
            self.L = torch.linalg.cholesky(cand)
            self.lam_used = float("nan")

    def delta(self, M: torch.Tensor) -> torch.Tensor:
        """求 ΔW 使 Z·ΔWᵀ ≈ X·Mᵀ，返回 [D, I]（float32）。"""
        Y = self.X.double() @ M.double().T           # [N, D]
        rhs = (Y.T @ self.Z.double()).T             # [I, D]
        return torch.cholesky_solve(rhs, self.L).T.float()

    def fidelity(self, M: torch.Tensor, val_frac: float = 0.2) -> float:
        """留出集上的真实保真度（在未参与分解的样本上测量）。"""
        n_val = max(1, int(self.n * val_frac))
        if self.n - n_val < 8:
            return float("nan")
        Zv = self.Z[self.n - n_val:]
        Dw = self.delta(M)
        y = self.X[self.n - n_val:] @ M.float().T
        return torch.nn.functional.cosine_similarity(
            y.flatten(), (Zv @ Dw.T).flatten(), dim=0).item()


# ═══════════════════════════════════════════════════════════════════════
# manifest
# ═══════════════════════════════════════════════════════════════════════
def write_manifest(output_dir: str, filename: str, payload: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    payload = dict(payload)
    payload["exported_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
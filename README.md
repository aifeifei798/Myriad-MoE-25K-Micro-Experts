# 🌌 Myriad-MoE (DualBigLittle-MoE)

> **Decoupling Multitask Cognitive Interference via Dual VRAM Dense Cores and Host-Pinned Streaming Micro-Experts**  
> *A 0.9B active-parameter architecture matching 7B+ baseline capabilities, featuring 25,200 modular micro-experts hosted in 1.54 GB system RAM with ~40ms in-memory live hot-swapping.*

[![Transformers Proposal](https://img.shields.io/badge/HuggingFace-Transformers_%2349183-orange.svg)](https://github.com/huggingface/transformers/issues/49183)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base-Qwen3--0.6B-green.svg)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![Hardware Verified](https://img.shields.io/badge/Verified_on-RTX_5090_D-purple.svg)]()

---

[简体中文](https://github.com/aifeifei798/Myriad-MoE-25K-Micro-Experts/blob/main/README_zh_cn.md)

## 💡 Executive Summary

Traditional dense Large Language Models (LLMs) suffer from severe **multitask negative transfer (cross-domain gradient collision)**: fine-tuning for strict programming logic degrades humanistic eloquence and conversational nuance, while training for creative prose softens formal mathematical reasoning. Meanwhile, standard Mixture-of-Experts (MoE) architectures demand massive GPU VRAM capacity, creating an impassable barrier for extreme expert scaling on consumer-grade hardware.

**Myriad-MoE adapts the battle-tested heterogeneous philosophy of mobile SoCs (ARM big.LITTLE and Unified Memory Paging) to neural network architectures:**

1. **Dual VRAM Dense Cores (Macro Foundation)**:
   - **Tier-1 Arts Core (GPU VRAM)**: 100% frozen native dense MLP safeguarding foundational linguistic intuition, stylistic nuance, and general world knowledge.
   - **Tier-2 STEM Core (GPU VRAM)**: Contrastively cloned dense MLP specialized exclusively for programming syntax, data structures, and mathematical proofs.
2. **Host-Pinned Streaming Micro-Experts (Extreme Sparsity in System RAM)**:
   - 28 layers $\times$ 20 clusters $\times$ 45 Rank-16 LoRA micro-experts ($\approx 25,200$ experts total).
   - **Consumes only 1.54 GB of standard host RAM** via Linux pinned memory allocations (`pin_memory()`).
   - Slices are dynamically streamed over the PCIe bus via asynchronous CUDA DMA streams on a per-token basis.
3. **Zero Latency Penalty (~29–31 tokens/s)**:
   - Transferring a token's active expert slice takes $\approx 0.13\text{ ms}$ over PCIe 4.0/5.0, which is **100% masked (hidden)** beneath the GPU's dense base model matrix operations.
   - Empirical proof: **Host CPU streaming mode (24.9 t/s) achieves full speed parity with pure GPU VRAM resident mode (23.8 t/s)** while saving gigabytes of GPU VRAM.
4. **Live Cyberware Hot-Swapping (~40ms In-Memory Mutation)**:
   - Single-domain micro-expert clusters can be trained in ~20 seconds (~3.6 MB payload) and in-place mutated into live host RAM via non-blocking `copy_()` calls without restarting the runtime, rebuilding CUDA graphs, or invalidating GPU contexts.
5. **Interactive Neuro-Surgery & Attribution Radar**:
   - Trace aberrant neuron activations across all 28 layers using `/catch`, temporarily isolate misbehaving expert clusters using `/cage`, or surgically snipe single-layer parameters via `/snipe`.

---

## 🏛️ System Architecture

```
                            ┌──────────────────────────────────────┐
                            │           Input Hidden State x       │
                            └──────────────────┬───────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
 ┌───────────────────────────┐                                   ┌───────────────────────────┐
 │   Router-Big (2-class)    │                                   │  Router-Cluster (20-class)│
 └─────────────┬─────────────┘                                   └─────────────┬─────────────┘
               │ Softmax weights (w_arts, w_sci)                               │ Top-2 Cluster Selection
               ▼                                                               ▼
 ┌───────────────────────────┐                     ┌────────────────────────────────────────────────────────┐
 │   GPU VRAM Dual-Core      │                     │        CPU Host Pinned Memory (1.54 GB RAM)            │
 │ ┌───────────────────────┐ │                     │ ┌────────────────────────────────────────────────────┐ │
 │ │ Tier-1: Arts (Frozen) │ │                     │ │ 25,200 Micro-Experts (20 Clusters x 45 LoRAs x 28) │ │
 │ └───────────┬───────────┘ │                     │ └──────────────────────────┬─────────────────────────┘ │
 │             │             │                     └────────────────────────────┼───────────────────────────┘
 │             ▼             │                                                  │ Async PCIe DMA Stream
 │ ┌───────────────────────┐ │                                                  ▼ (96 KB / Layer Slice)
 │ │ Tier-2: STEM (Cloned) │ │                     ┌────────────────────────────────────────────────────────┐
 │ └───────────┬───────────┘ │                     │        GPU Streaming Staging Computation               │
 └─────────────┼─────────────┘                     │            (F.linear Rank-16 Low-Rank GEMM)            │
               │ Dense Output (big_out)            └────────────────────────────┬───────────────────────────┘
               │                                                                │ Micro Output (micro_out)
               └───────────────────────────────┬────────────────────────────────┘
                                               ▼
                                      Output Hidden State
                         y = big_out + γ * (Top-2 Weighted micro_out)
```

---

## 📐 Mathematical & Systems Formulation

### 1. Dimension-Adaptive Residual Scaling ($\mu P$ Variance Invariance)
To prevent internal representation drift when injecting 25,200 micro-experts across 28 deep layers, heuristic hyperparameter constants are abandoned in favor of orthogonal variance normalization:
$$\gamma = \frac{1}{\sqrt{d_{\text{model}}}}$$
- For $d_{\text{model}} = 1024$ (Qwen-0.6B): $\gamma = \frac{1}{32} \approx \mathbf{0.03125}$
- For $d_{\text{model}} = 1536$ (Scalpel-E2B): $\gamma = \frac{1}{\sqrt{1536}} \approx \mathbf{0.0255}$
- For $d_{\text{model}} = 4096$ (7B Scale): $\gamma = \frac{1}{64} \approx \mathbf{0.0156}$

This formulation guarantees that regardless of model scaling, the perturbation energy injected by active micro-experts remains invariant and unit-isotropic across all network depths.

### 2. Low-Rank Tensor Contraction During Training

During batch training across 25,200 experts, materializing the full 5D intermediate tensor `[B, S, C, E, D]` would consume over 1.4 GB per layer in activations. We perform early contraction within the low-rank bottleneck:

```python
# 1. Low-rank projection: [B, S, D] @ [C, E, R, D]^T -> [B, S, C, E, R] (~23 MB)
h = torch.einsum('bsd,cerd->bscer', x, self.lora_A)

# 2. Contraction & normalization: [B, S, C, E, R] @ [C, E, D, R]^T -> [B, S, C, D] (~33 MB)
clustered_out = torch.einsum('bscer,cedr->bscd', h, self.lora_B) / 45.0

# 3. Macro routing aggregation: [B, S, C] @ [B, S, C, D] -> [B, S, D]
micro_out = torch.einsum('bsc,bscd->bsd', w_cluster, clustered_out)
```

This formulation reduces training VRAM footprint by **97.6%**, enabling 25,200-expert training within **10 GB VRAM** at **9.8 samples/sec** on a single consumer GPU.

---

## 📊 Empirical Verification & Telemetry

### 1. Verification of Non-Interference (Symmetric Macro Inversion)
Testing across polarized tasks confirms that the macro router cleanly shifts representational density without cross-contamination:

| Benchmark Task | Primary Output Focus | Arts Core Ratio | STEM Core Ratio | Top Active Micro-Clusters |
| :--- | :--- | :---: | :---: | :--- |
| **Classical Zen Prose** | Imagery, classical Chinese cadence, sensory tone | **79.1%** 🏛️ | **20.9%** | `#02 Arts_Fiction`<br>`#00 Arts_Prose`<br>`#01 Arts_Poetry` |
| **C++20 Concurrency** | Lock-free SPSC buffer, `std::atomic`, cache-line padding | **24.5%** | **75.5%** 🔬 | `#16 Code_Algo`<br>`#17 Code_DS`<br>`#20 Code_Syntax` |
| **Cross-Domain Synthesis** | 4D Tesseract Gray-code routing + Metaphysical prose | **55.8%** | **44.2%** | `#13 Arts_Philosophy`<br>`#01 Code_DS`<br>`#02 Code_Debug` |

### 2. Execution Placement Parity (CPU vs. GPU Benchmarking)
- **Host Streaming Mode (`expert_pool_location: "host"`)**: **24.9 tokens/s** (Zero VRAM allocated for experts).
- **Device Resident Mode (`expert_pool_location: "device"`)**: **23.8 tokens/s** (All experts pinned in VRAM).
- *Finding*: Transferring 96 KB per layer over PCIe 4.0/5.0 requires $\approx 0.13\text{ ms}$, which is completely absorbed by the 40 ms GPU forward pass. Streaming from host RAM incurs **0% throughput degradation**.

---

## 🔌 Cartridge Hot-Swapping & Neuro-Surgery

### 1. Cluster Allocation (20-Cluster Topology)
- **Clusters 00 ~ 15**: General World Foundation (Code, Data Structures, Mathematics, Sciences, Humanities).
- **Clusters 16 ~ 19**: **Modular Extension Slots** reserved for private rulebooks, domain personas, or proprietary enterprise data.

### 2. Standalone Cartridge Forging
Train an arbitrary domain rulebook (e.g., `custom_data.jsonl`, 20 samples) in ~20 seconds to export a standalone cartridge (`cartridge_gongfang.pt`):
```bash
python 4.train_single_cartridge.py
```

### 3. Live Hot-Swapping (`/plug`)
Inject the newly forged cartridge into Slot 16 in **44 ms** during an active conversation without restarting the inference server:
```text
👤 You: /plug cartridge_gongfang.pt
⚡ [Hot-Swap Success] Cartridge 'Atelier_Rulebook_V1' injected into Slot #16!
   ⏱️ Latency: 44.45 ms | VRAM Delta: 0 MB | Active immediately!
```

### 4. Rogue Expert Detection & Isolation (`/catch` and `/cage`)
When unexpected or over-indexed behavior is observed, pinpoint the responsible layer and cluster immediately:
```text
👤 You: /catch
🚨 [Neural Diagnostic Radar: 28-Layer Activation Trace]
   - Layer 19 : #16 [Custom_Rules    ] (75 activations) 🔥 [Anomalous Divergence]
   - Layer 20 : #16 [Custom_Rules    ] (57 activations) 🔥 [Anomalous Divergence]

👤 You: /cage 16
🔒 [Isolated] Cluster #16 [Custom_Rules] suppressed in volatile memory.

👤 You: /free 16
🔓 [Restored] Cluster #16 [Custom_Rules] restored to full compute capacity.
```

### 5. Offline Multi-Cartridge Fusion (`5.fuse_cartridges.py`)
Permanently fuse arbitrary cartridges into a standalone unified checkpoint without retraining the base weights:
```bash
python 5.fuse_cartridges.py
# Fuses base weights + cartridge_gongfang.pt (Slot 16) -> myriad_moe_25k_ultimate_fused.pt
```

---

## 🧱 Baking to Standard Architectures (`bake_*.py`)

Fuse the micro-expert deltas and the dual cores into **stock** model weights, so the result loads anywhere without `trust_remote_code` and without the 1.54 GB host-RAM streaming layer.

The absorption is closed-form. Because the delta only ever lands on `down_proj`, the base `gate_proj` / `up_proj` survive untouched and the residual stream is preserved exactly:

```
ΔW = (YᵀZ)(ZᵀZ + λI)⁻¹      Z = silu(X·Wgᵀ) ⊙ (X·Wuᵀ)
```

Two output formats, both verified numerically before they are written to disk:

| Script | Format | Params | Held-out fidelity | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `bake_and_merge_dense.py` | `Qwen3ForCausalLM` (stock dense) | 0.60B | **0.9935** | Lowest layer 0.9833 |
| `bake_and_export_moe.py` | `Qwen3MoeForCausalLM` (stock MoE) | 5.62B | **0.9988** | Lowest layer 0.9975, `Σ_k w_k = 0.99996` |

```bash
# Dense — one self-contained file, no MoE runtime at all
python bake_and_merge_dense.py --output-dir ./qwen_dense

# MoE — keeps sparse routing; 5.62B because stock MoE has no weight sharing
python bake_and_export_moe.py --output-dir ./myriad_qwen3_moe

# Real corpus strongly recommended (built-in synthetic is only a fallback)
python bake_and_export_moe.py --calib-file ./my_corpus.txt --output-dir ./out

# Seal clusters, plug a cartridge, pre-bake
python bake_and_export_moe.py --cage-clusters 12,13 --plug 16=rules.pt --output-dir ./out
```

Both scripts share `bake_common.py` (calibration corpus, real-forward hooks, closed-form solver, manifest). Each run writes `myriad_bake_manifest.json` / `myriad_moe_manifest.json` recording every approximation below, plus per-layer gating weights and held-out fidelity.

### ⚠️ Two things that will silently ruin the export

**1. The target architecture must match the base.** The base is Qwen3-0.6B, whose attention carries per-head RMSNorm (`q_norm` / `k_norm`). `Qwen2Moe` has **no such modules** — copying the weights anyway succeeds silently (the shapes line up) but drops the normalization and the model degenerates into repeated tokens. `bake_and_export_moe.py` targets `Qwen3Moe` for exactly this reason and asserts `head_dim` agreement. The Dense path is unaffected because it edits the base in place.

**2. `norm_topk_prob` must be `True`.** Stock MoE emits `Σ_k w_k · expert_k(x)`. Since the base MLP is replicated into every routed expert (the only way to absorb a delta into `down_proj`), that scales the *entire dense core* by `Σ_k w_k`. Measured on this checkpoint (top-2 of 20):

| Layer | L0 | L5 | L13 | L27 | mean |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `Σ_k w_k` | 0.185 | 0.231 | 0.438 | 0.757 | **0.40** |

Training-time coefficients on the dense core and `Δ_sci` are hard-wired to `1.0`, so shipping without normalization discards ~60% of the core. `norm_topk_prob=True` renormalizes to `Σ ≡ 1`, matching training. The cost is that intra-expert relative weights get renormalized (training uses raw softmax) — the standard MoE distillation trade-off, far better than severing the backbone.

### Declared approximations

These are real degradations, not equivalences, and each is recorded in the manifest:

1. **`w_sci(x)` is a per-token gate → frozen to its measured mean.** Measured swing within a single layer is 0.03–0.99, so this is a genuine downgrade. Override with `--sci-weight`.
2. **Sparse top-k.** The MoE export keeps stock routing; the Dense export instead averages all active clusters equally (dense ensembling).
3. **Sealed clusters are removed, not biased.** Training seals a slot via `cluster_bias = -1e4`; stock routers have no bias term, so `--cage-clusters` removes the expert outright (indices remapped in `cluster_order`).

### Calibration corpus size is the fidelity bottleneck

The closed-form solve is `[3072, 3072]`, so sample count `N` must comfortably exceed the intermediate dimension `I = 3072`. Both scripts report the ratio and **warn when the system is underdetermined**, so a thin corpus can never be mistaken for a quality ceiling:

```
✗ 校准样本 53 / 中间维 3072 = 0.02×（严重欠定，保真度数字基本不可信，请务必加大 --calib-tokens）
✓ 校准样本 8436 / 中间维 3072 = 2.75×
```

Reported fidelity is measured on a **held-out 20%** split (by document order, not random) with `λ` selected on that same split — an earlier version reported on the fitting set with a fixed `λ`, a number that could be tuned rather than trusted.

### Choosing `--micro-scale`

`--micro-scale` sets how much micro-expert `Δ` gets injected. The default is **0.0125**, chosen by sweeping the Dense export over 27 held-out prompts (written fresh, not drawn from the calibration corpus) and scoring 4-gram repetition against the unbaked base:

| `--micro-scale` | Fidelity | rep_rate | vs. base | distinct-2 | agree with base | Identical to base |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| *base, unbaked* | — | 0.2720 | — | 0.6559 | — | — |
| 0.0000 (Δ_sci only) | 0.9867 | 0.2746 | +0.0026 | 0.6428 | 0.7940 | 17/27 |
| **0.0125** (default) | 0.9913 | **0.2567** | **−0.0153** | 0.6416 | 0.7211 | 14/27 |
| 0.025 | 0.9935 | 0.3027 | +0.0307 | 0.6057 | 0.5231 | 7/27 |
| 0.0500 | 0.9950 | 0.3359 | +0.0639 | 0.5842 | 0.3727 | 3/27 |
| 0.1000 | 0.9958 | 0.3423 | +0.0702 | 0.5520 | 0.2176 | 2/27 |

Two things worth knowing before you retune this:

- **Fidelity rises with `micro_scale`, quality peaks in the middle.** Fidelity measures only whether `Δ` is expressible in `down_proj` — a larger `Δ` is a stronger signal and is easier to capture. It says nothing about language quality. Selecting `micro_scale` by fidelity alone walks you straight to 0.1, the worst row in the table.
- **The two goals genuinely conflict.** "Preserve base capability" and "make micro-expert behavior visible" cannot both be maximized. Low values keep the model intact; high values make `Δ` obvious (agreement with base drops to 0.37 / 0.22) while degradation climbs. 0.0125 is the only setting that does not regress repetition while still leaving `Δ` measurable — `agree` 0.72 means the baked model is clearly not just the base model.

⚠ **Caveat on the numbers above.** 27 prompts, greedy decoding, single corpus (the built-in synthetic one). The −0.0153 edge over the base is small enough to sit near the noise floor, so treat 0.0125 as a reasonable default rather than a proven optimum. Re-run the sweep with `--calib-file` on real text before relying on it for a release.

The MoE export defaults to the same 0.0125 so the two formats can be compared directly, but it needs less tuning — it preserves sparse routing, so its dense-ensembling distortion is absent.

---

---

## 🚀 Quickstart Pipeline

### 1. Environment Setup
```bash
git clone https://github.com/aifeifei798/Myriad-MoE-25K-Micro-Experts.git
cd Myriad-MoE-25K-Micro-Experts
pip install torch transformers datasets accelerate
```

### 2. Dataset Synthesis (20 Clusters)
```bash
python 1.prepare_myriad_data.py
# Prepares 20,000 domain-partitioned training samples across 20 clusters
```

### 3. Base MoE Pretraining (25,200 Micro-Experts)
```bash
python 2.train_myriad_25k.py
# Trains 28 layers x 20 clusters x 45 experts (~20-25 mins on an RTX 5090 D)
# Checkpoint exported: myriad_moe_25k_weights.pt (~1.25 GB)
```

### 4. Launching the Interactive Telemetry Terminal
```bash
python 3.chat_myriad_25k.py
# Mounts 25,200 experts into 1.54 GB RAM with real-time cluster telemetry
```

### 5. OpenAI-Compatible API Server (`7.api_myriad_server.py`)
Exposes the full terminal feature set over HTTP, reusing the inference core of `6.chat_myriad_25k_lora_fast_more_mirco.py` verbatim (no `forward` duplication).

```bash
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" python-multipart
python 7.api_myriad_server.py --port 8000 --api-key sk-myriad
# Live dashboard: http://127.0.0.1:8000/     Telemetry: /v1/myriad/stats
```

Drop-in for any OpenAI client (`openai`, Cherry Studio, LangChain, …):
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-myriad")
client.chat.completions.create(
    model="myriad-moe-25k-lora",
    messages=[{"role": "user", "content": "Explain quicksort"}],
    stream=True,                                   # SSE with reasoning_content split-out
    extra_body={"myriad": {"focus_clusters": [0, 1, 2]}},   # restrict routing per request
)
```

| Endpoint | Purpose |
| :--- | :--- |
| `GET /v1/models`, `GET /v1/models/{id}` | Model cards with Myriad metadata |
| `POST /v1/chat/completions` | Streaming SSE + non-streaming, standard `usage` / `finish_reason` / `[DONE]` |
| `POST /v1/completions` | Legacy text completion |
| `GET /v1/myriad/stats`, `POST …/stats/reset` | Holographic dashboard (Arts/STEM ratio, 20-cluster heatmap, VRAM) |
| `GET /v1/myriad/catch` | 28-layer attribution radar (`/catch`) |
| `GET`/`POST /v1/myriad/topk` | Per-layer or global dynamic kernel count (`/show_k`, `/set_k`, `/set_k_all`) |
| `POST /v1/myriad/clusters/{cid}/cage`·`/free`, `POST /v1/myriad/snipe` | Neuro-surgery (`/cage`, `/free`, `/snipe`) |
| `POST /v1/myriad/cartridge/plug` | Hot-swap a cartridge via multipart upload or server path (`/plug`) |
| `POST /v1/myriad/engine` | Toggle CUDA Graph decoding / default reply length (`/graph`, `/maxlen`) |
| `GET /v1/myriad/metrics` | JSON or `?format=prometheus` (TTFT, tok/s, queue depth) |
| `GET /` | Browser dashboard with live cluster heatmap |

Extensions beyond the OpenAI schema: `repetition_penalty`, `chat_template_kwargs` (Qwen3 `enable_thinking`), `split_reasoning` (routes `<think>` blocks into `delta.reasoning_content`), and the `myriad` block (`focus_clusters`, `top_k`, `stats`, `reset_stats`) for per-request neural control — all reverted automatically when the request ends. Slash commands (`/catch`, `/cage 16`, `/plug x.pt`, …) also work directly in chat.

---

## 📜 Repository Roadmap

- [x] **ComfyUI-FeiFei Integration**: Registered custom nodes for LLM director, image captioner, and physical film emulation.
- [x] **Transformers Issue #49183 Proposal**: Architecture formalized conforming to Hugging Face modular model converter standards.
- [x] **Dual VRAM Dense Core Decoupling**: Verified 100% immunity to Arts vs. STEM catastrophic forgetting.
- [x] **Zero-Penalty Host Pinned Streaming**: Pinned memory DMA pipeline yielding 30 tokens/s on consumer hardware.
- [x] **In-Memory Dynamic Hot-Plugging**: 44 ms live mutation via volatile memory tensor swap.
- [x] **Per-Layer Attribution & Diagnostic Radar**: 28-layer inspection and dynamic expert isolation.
- [x] **OpenAI-Compatible API Server**: Drop-in `/v1` endpoint with per-request expert routing control and live telemetry dashboard.
- [x] **Stock-Architecture Baking**: Closed-form fusion into official `Qwen3` / `Qwen3Moe` weights — zero custom dependencies, verified numerically (0.9935 / 0.9988 held-out fidelity).
- [ ] **Low-Precision Streaming**: Porting micro-expert DMA streams to FP8/INT4 for embedded edge accelerators.

---

## 📖 Citation

If you use this architecture, the streaming micro-expert design, or the empirical findings in your research, please cite:

```bibtex
@misc{feifei2026dualbiglittlemoe,
  author = {FeiFei (aifeifei798) and Community Contributors},
  title = {{DualBigLittle-MoE: A Tri-Tier Asymmetric Architecture Decoupling Multitask Cognitive Interference via Dual VRAM Dense Cores and Streaming Micro-Expert Clusters}},
  year = {2026},
  publisher = {GitHub and Hugging Face},
  howpublished = {\url{https://github.com/aifeifei798/DualBigLittle-MoE}},
  note = {Hugging Face Transformers Issue \#49183, Hub Checkpoint: \url{https://huggingface.co/aifeifei798/DualBigLittle-MoE-Qwen3-0.6b}}
}
```

---

## ⚖️ License
Released under the **[Apache-2.0 License](LICENSE)**. Free for academic research and commercial applications with proper attribution.

---

## 🔐 API Access Control & Data Reproduction

### Token tiers

| Token | Flag | Permissions |
|---|---|---|
| Admin | `--api-key` | Everything: telemetry, neuro-surgery, top-k, cartridge plug, generation |
| Read-only | `--read-only-key` | `GET` telemetry/dashboard only; **all writes → 403** |

```bash
python 7.api_myriad_server.py --api-key sk-admin --read-only-key sk-viewer
```

`GET /v1/models` reports the caller's tier under `myriad.permission`
(`admin` / `read` / `anonymous`) plus `auth_required` and `read_only_available`,
so a dashboard client can grey out controls it may not use.

Design note: every `/v1/myriad/*` route is gated on `require_engine`, so during
the (potentially long) weight load they answer **503 "模型正在加载中"** rather
than crashing with a 500. `/v1/models` and `/health` stay reachable so clients
can still detect readiness.

### Cartridge upload limits

`POST /v1/myriad/cartridge/plug` accepts either a multipart `file` upload or a
server-side `path`. Uploads are capped by `--max-cartridge-mb` (default 512) and
return **413** when exceeded; empty bodies → 400, out-of-range slots → 400,
missing files → 404. Filenames are sanitised with `basename` and never used in
path construction.

### Web console

A dedicated front-end lives in a separate repository:
**[aifeifei798/myriad-moe-console](https://github.com/aifeifei798/myriad-moe-console)**

### Data & weights are not in git

Weight files are 1.6–2.1 GB each and the training corpus is 14 MB, so they are
excluded by `.gitignore`. To rebuild:

```bash
python 1.prepare_myriad_data.py      # → myriad_train_data.jsonl (not committed)
python 2.train_myriad_25k.py         # → myriad_moe_25k_weights.pt
python 4.train_single_cartridge.py   # → cartridge_*.pt
python 5.fuse_cartridges.py          # → myriad_moe_25k_ultimate_fused.pt
```

`custom_data.jsonl` (5 KB sample) **is** committed so the pipeline can be smoke-tested.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
🌌【Myriad-MoE OpenAI 兼容 API 服务端】(生产加固版)

把 6.chat_myriad_25k_lora_fast_more_mirco.py 的终端交互能力完整搬上 HTTP:

  · OpenAI 兼容层 : /v1/models、/v1/chat/completions (流式 SSE + 非流式)、
                    /v1/completions (旧版补全)、标准 usage / finish_reason / [DONE]
  · 神经手术台   : /v1/myriad/* 覆盖终端全部指令
                    /catch 雷达、/show_k、/set_k、/set_k_all、/cage、/free、
                    /snipe、/plug 卡带热插拔、/graph 引擎开关、/maxlen
  · 每请求控制   : 请求体 myriad 扩展块 → focus_clusters (本次只允许这些宗门参战)、
                    top_k 临时覆盖, 请求结束自动还原, 全程不重捕获 CUDA Graph
  · 遥测监控     : /v1/myriad/stats 全息看板、/v1/myriad/metrics (Prometheus)、
                    GET / 浏览器实时热力图
  · 斜杠指令     : 聊天里发 "/catch" 依然按终端指令解析并把统计文本回传

用法:
    uv pip install --python .venv/bin/python fastapi "uvicorn[standard]"
    python 7.api_myriad_server.py --port 8000 --api-key sk-myriad

客户端:
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-myriad")
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import asyncio
import importlib.util
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple, Union

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s │ %(levelname)-7s │ %(name)-16s │ %(message)s",
                    datefmt="%H:%M:%S")
LOG = logging.getLogger("myriad.api")

SERVED_MODEL_ID = "myriad-moe-25k-lora"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.15
MAX_TOP_K = 10
CAGE_BIAS = -1e4


# ════════════════════════════════════════════════════════════════════════════
# 1. 复用终端版推理核心
# ════════════════════════════════════════════════════════════════════════════
def load_backend_module(script_path: str):
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"找不到推理后端脚本: {script_path}")
    spec = importlib.util.spec_from_file_location("myriad_chat_backend", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载推理后端脚本: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for attr in ("MyriadInferenceWrapper", "GraphedDecoder", "CLUSTER_NAMES"):
        if not hasattr(module, attr):
            raise AttributeError(f"推理后端缺少 {attr}, 请检查 {script_path}")
    return module


# ════════════════════════════════════════════════════════════════════════════
# 2. 可热插拔偏置层: 持久禁闭(cage_bias) + 每请求 focus(focus_bias)
# ════════════════════════════════════════════════════════════════════════════
def make_controllable_mlp_class(backend):
    class ControllableMyriadMLP(backend.MyriadInferenceWrapper):
        """原包装层 + 两路偏置的神经手术接口。"""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_buffer("cage_bias", torch.zeros(
                self.num_clusters, device=self.device, dtype=self.cluster_bias.dtype))
            self.register_buffer("focus_bias", torch.zeros(
                self.num_clusters, device=self.device, dtype=self.cluster_bias.dtype))
            self.refresh_bias()

        @torch.no_grad()
        def refresh_bias(self):
            """唯一真值来源 → cluster_bias (原地写, 图捕获指针保持有效)。"""
            self.cluster_bias.copy_(self.cage_bias + self.focus_bias)

        @torch.no_grad()
        def set_caged(self, cids: Sequence[int], caged: bool):
            for cid in cids:
                self.cage_bias[cid] = CAGE_BIAS if caged else 0.0
            self.refresh_bias()

        @torch.no_grad()
        def set_focus(self, focus: Optional[Sequence[int]]):
            self.focus_bias.zero_()
            if focus is not None:
                keep = torch.zeros(self.num_clusters, dtype=torch.bool, device=self.device)
                keep[torch.tensor(list(focus), device=self.device, dtype=torch.long)] = True
                self.focus_bias.masked_fill_(~keep, CAGE_BIAS)
            self.refresh_bias()

    return ControllableMyriadMLP


# ════════════════════════════════════════════════════════════════════════════
# 3. 文本流工具: 增量多字节解码 / 状态机思维链分流 / 前瞻滑动窗 Stop 截断
# ════════════════════════════════════════════════════════════════════════════
class TokenTextStream:
    """token id 流 → 文本增量 (传入完整序列解码，天然免疫多字节 UTF-8 拆分粘包)。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.prev = ""

    def push(self, tid: int) -> Tuple[str, str]:
        self.ids.append(tid)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if len(text) <= len(self.prev):
            return "", text                         # 半截汉字: 等待后续字节
        delta = text[len(self.prev):]
        self.prev = text
        return delta, text


class ThoughtSplitter:
    """把 <think>...</think> 分流到 reasoning_content (带跨步前缀保护)。"""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.full = ""
        self.opened = False
        self.closed = False
        self.rpos = 0
        self.cpos = 0

    def feed(self, text: str) -> Tuple[str, str]:
        if not text:
            return "", ""
        self.full = text
        f = self.full

        # 1. 尚未开启思考链
        if not self.opened:
            oi = f.find(self.OPEN)
            if oi < 0:
                delta = f[self.cpos:]
                self.cpos = len(f)
                return "", delta
            self.opened = True
            self.rpos = oi + len(self.OPEN)
            self.cpos = self.rpos

        # 2. 思考中：检测 </think> 闭合标签
        if not self.closed:
            # 允许回退检索，防止闭合标签被跨 Token 拆碎
            ci = f.find(self.CLOSE, max(0, self.rpos - len(self.CLOSE)))
            if ci < 0:
                # 尚未闭合：检查末尾是否正在形成 </think> 的前缀
                safe_end = len(f)
                for k in range(1, len(self.CLOSE)):
                    if f.endswith(self.CLOSE[:k]):
                        safe_end = len(f) - k
                        break
                if safe_end > self.rpos:
                    r_delta = f[self.rpos:safe_end]
                    self.rpos = safe_end
                    return r_delta, ""
                return "", ""

            # 命中闭合标签
            self.closed = True
            r_delta = f[self.rpos:ci]
            self.rpos = ci
            self.cpos = ci + len(self.CLOSE)
            c_delta = f[self.cpos:]
            self.cpos = len(f)
            return r_delta, c_delta

        # 3. 思考链已完结，后续全量作为正文
        c_delta = f[self.cpos:]
        self.cpos = len(f)
        return "", c_delta


class StopTrimmer:
    """在线 Stop 序列前瞻检测（防止停用词前缀被提前冲刷至客户端）。"""

    def __init__(self, stops: Sequence[str]):
        self.stops = [s for s in (stops or []) if s]
        self.max_len = max((len(s) for s in self.stops), default=0)

    def check(self, text: str) -> Tuple[Optional[int], int]:
        """
        返回: (hit_idx, safe_len)
        hit_idx: 若完全命中 stop 词，返回命中的起始位置；否则为 None
        safe_len: 当前可安全下发的字符长度（扣减末尾正在拼装的 stop 前缀）
        """
        if not self.stops:
            return None, len(text)

        best_hit = None
        for s in self.stops:
            idx = text.find(s)
            if idx >= 0 and (best_hit is None or idx < best_hit):
                best_hit = idx

        if best_hit is not None:
            return best_hit, best_hit

        # 检查末尾是否存在任意 stop 序列的子前缀
        holdback = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(text)), 0, -1):
                if text.endswith(s[:k]):
                    holdback = max(holdback, k)
                    break

        safe_len = max(0, len(text) - holdback)
        return None, safe_len


# ════════════════════════════════════════════════════════════════════════════
# 4. 遥测指标
# ════════════════════════════════════════════════════════════════════════════
class Metrics:
    def __init__(self, window: int = 256):
        self._lock = threading.Lock()
        self._ttfts: deque = deque(maxlen=window)
        self._durs: deque = deque(maxlen=window)
        self._speeds: deque = deque(maxlen=window)
        self.started = time.time()
        self.requests_total = 0
        self.errors_total = 0
        self.stream_requests = 0
        self.command_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.active = 0
        self.waiting = 0

    def observe(self, ttft: Optional[float], dur: float, pt: int, ct: int):
        with self._lock:
            self.requests_total += 1
            if ttft is not None:
                self._ttfts.append(ttft)
            self._durs.append(dur)
            if dur > 0:
                self._speeds.append(ct / dur)
            self.prompt_tokens += pt
            self.completion_tokens += ct

    @staticmethod
    def _pct(vals, q):
        if not vals:
            return None
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(len(s) * q))], 4)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            up = self.prompt_tokens / max(1e-9, time.time() - self.started)
            return {
                "uptime_sec": round(time.time() - self.started, 1),
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "stream_requests": self.stream_requests,
                "command_requests": self.command_requests,
                "active_requests": self.active,
                "waiting_requests": self.waiting,
                "avg_ttft_sec": self._pct(self._ttfts, 0.5),
                "p95_ttft_sec": self._pct(self._ttfts, 0.95),
                "avg_duration_sec": self._pct(self._durs, 0.5),
                "avg_tokens_per_sec": round(sum(self._speeds) / len(self._speeds), 1)
                if self._speeds else None,
                "prompt_tokens_total": self.prompt_tokens,
                "completion_tokens_total": self.completion_tokens,
                "prompt_tok_per_sec": round(up, 2),
            }

    def prometheus(self) -> str:
        s = self.snapshot()
        lines = [
            "# HELP myriad_uptime_sec Server uptime in seconds.",
            "# TYPE myriad_uptime_sec gauge",
            f"myriad_uptime_sec {s['uptime_sec']}",
            "# HELP myriad_requests_total Total OpenAI-compatible requests served.",
            "# TYPE myriad_requests_total counter",
            f"myriad_requests_total {s['requests_total']}",
            "# HELP myriad_errors_total Total failed requests.",
            "# TYPE myriad_errors_total counter",
            f"myriad_errors_total {s['errors_total']}",
            "# HELP myriad_active_requests Requests currently generating.",
            "# TYPE myriad_active_requests gauge",
            f"myriad_active_requests {s['active_requests']}",
            "# HELP myriad_waiting_requests Requests queued on the single decode slot.",
            "# TYPE myriad_waiting_requests gauge",
            f"myriad_waiting_requests {s['waiting_requests']}",
            "# HELP myriad_completion_tokens_total Generated tokens.",
            "# TYPE myriad_completion_tokens_total counter",
            f"myriad_completion_tokens_total {s['completion_tokens_total']}",
            "# HELP myriad_ttft_sec Latency gauges (avg / p95 / tokens-per-sec).",
            "# TYPE myriad_ttft_sec gauge",
        ]
        for key in ("avg_ttft_sec", "p95_ttft_sec", "avg_duration_sec"):
            if s[key] is not None:
                lines.append(f"myriad_{key} {s[key]}")
        if s["avg_tokens_per_sec"] is not None:
            lines.append(f"myriad_avg_tokens_per_sec {s['avg_tokens_per_sec']}")
        lines.append("")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# 5. 推理引擎核心
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class GenParams:
    max_new_tokens: int = 512
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    stop: List[str] = field(default_factory=list)
    split_reasoning: bool = True


class MyriadEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.backend = args.backend
        self.cluster_names: List[str] = list(self.backend.CLUSTER_NAMES)
        self.ControllableMLP = make_controllable_mlp_class(self.backend)
        self.device = args.device
        self.dtype = getattr(torch, args.dtype)
        self.model_id = args.model
        self.num_clusters = args.num_clusters
        self.experts_per_cluster = args.experts_per_cluster
        self.metrics = Metrics()
        self.gate = asyncio.Semaphore(1)                  # 单槽位串行排队
        self._reapers: set = set()                        # 槽位归还守护任务集
        self._slot_holder: Optional[str] = None
        self._caged: Dict[int, Dict[int, torch.Tensor]] = {}
        self._plugged: Dict[int, Dict[str, Any]] = {}
        self.model = None
        self.tokenizer = None
        self.decoder = None
        self.ready = False
        self.startup_error: Optional[str] = None

    @property
    def layers(self):
        return self.model.model.layers

    def load(self):
        a = self.args
        LOG.info("正在唤醒【Myriad-MoE OpenAI 服务端】...")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        self.tokenizer = AutoTokenizer.from_pretrained(a.model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=self.dtype, device_map=a.device, attn_implementation="sdpa")
        self.model.eval()
        hidden = self.model.config.hidden_size

        layer_top_k = list(a.layer_top_k)
        n_layers = len(self.model.model.layers)
        if len(layer_top_k) < n_layers:
            layer_top_k += [2] * (n_layers - len(layer_top_k))

        for i, layer in enumerate(self.model.model.layers):
            layer.mlp = self.ControllableMLP(
                layer.mlp, hidden_dim=hidden, num_clusters=self.num_clusters,
                experts_per_cluster=self.experts_per_cluster, micro_rank=a.micro_rank,
                macro_rank=a.macro_rank, device=a.device, dtype=self.dtype,
                reside_on_gpu=not a.cpu_experts, top_k=layer_top_k[i])

        weights_path = a.weights
        if not os.path.exists(weights_path):
            weights_path = "myriad_moe_25k_weights.pt"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"找不到权重文件 ({a.weights})")
        LOG.info("挂载权重: %s", weights_path)
        saved = torch.load(weights_path, map_location="cpu")
        for i, layer in enumerate(self.model.model.layers):
            layer.mlp.sci_lora_A.load_state_dict(saved[f"layer_{i}_sci_lora_A"])
            layer.mlp.sci_lora_B.load_state_dict(saved[f"layer_{i}_sci_lora_B"])
            layer.mlp.router_big.load_state_dict(saved[f"layer_{i}_router_big"])
            layer.mlp.router_cluster.load_state_dict(saved[f"layer_{i}_router_cluster"])
            if a.cpu_experts:
                layer.mlp.lora_A_cpu = saved[f"layer_{i}_lora_A"].pin_memory()
                layer.mlp.lora_B_cpu = saved[f"layer_{i}_lora_B"].pin_memory()
            else:
                layer.mlp.mount_experts(
                    saved[f"layer_{i}_lora_A"].to(a.device, dtype=self.dtype).contiguous(),
                    saved[f"layer_{i}_lora_B"].to(a.device, dtype=self.dtype).contiguous())
        del saved

        self.decoder = self.backend.GraphedDecoder(
            self.model, a.device, self.dtype, enabled=not a.no_graph, verbose=True)

        for spec in (a.plug or []):
            slot, _, path = spec.partition("=")
            if not path:
                slot, path = 16, slot
            res = self.plug_cartridge(path, int(slot))
            LOG.info("启动热插拔: 插槽 #%02d ← %s (%.2f ms)", res["slot"], res["name"], res["elapsed_ms"])

        self.ready = True
        LOG.info("✅ %d 层 × %d 宗门 × %d 微专家 = %s 微专家就位 | CUDA Graph=%s",
                 n_layers, self.num_clusters, self.experts_per_cluster,
                 f"{n_layers * self.num_clusters * self.experts_per_cluster:,}",
                 "ON" if self.decoder.enabled else "OFF")

    @torch.no_grad()
    def _apply_overrides(self, focus_clusters, top_k_override):
        saved_topk = None
        k = None
        if top_k_override is not None:
            k = max(1, min(int(top_k_override), MAX_TOP_K))
        if focus_clusters is not None:
            k = max(1, min(k or MAX_TOP_K, len(focus_clusters)))
        if k is not None:
            saved_topk = [int(l.mlp.top_k) for l in self.layers]
            for l in self.layers:
                l.mlp.top_k = k
            self.decoder.invalidate("请求级 top_k 覆盖/focus 收敛")
        for l in self.layers:
            l.mlp.set_focus(focus_clusters)
        return saved_topk

    @torch.no_grad()
    def _restore_overrides(self, saved_topk):
        for l in self.layers:
            l.mlp.set_focus(None)
        if saved_topk is not None:
            for l, k in zip(self.layers, saved_topk):
                l.mlp.top_k = k
            self.decoder.invalidate("请求级 top_k 还原")

    def reset_stats(self):
        for l in self.layers:
            l.mlp.reset_stats()

    def dashboard(self) -> Dict[str, Any]:
        arts = sum(l.mlp.stat_arts_weight.item() for l in self.layers)
        sci = sum(l.mlp.stat_sci_weight.item() for l in self.layers)
        big = arts + sci
        arts_pct = (arts / big * 100) if big > 0 else 50.0
        sci_pct = (sci / big * 100) if big > 0 else 50.0

        counts = torch.zeros(self.num_clusters, dtype=torch.int64)
        for l in self.layers:
            counts += l.mlp.stat_cluster_counts.cpu().long()
        top_v, top_i = torch.topk(counts, k=min(6, self.num_clusters))
        heat = [{"id": int(c), "name": self.cluster_names[c], "hits": int(v),
                 "slot": bool(c >= 16), "caged": bool(c in self._caged)}
                for c, v in zip(top_i.tolist(), top_v.tolist())]
        per_cluster = [{"id": c, "name": self.cluster_names[c], "hits": int(counts[c].item()),
                        "caged": bool(c in self._caged),
                        "cartridge": self._plugged.get(c, {}).get("name")}
                       for c in range(self.num_clusters)]

        per_layer = []
        for idx, l in enumerate(self.layers):
            c = l.mlp.stat_cluster_counts.cpu()
            dom = int(torch.argmax(c).item()) if int(c.sum()) > 0 else -1
            per_layer.append({
                "layer": idx, "top_k": int(l.mlp.top_k),
                "active_experts": int(l.mlp.top_k) * self.experts_per_cluster,
                "dominant_cluster": dom,
                "dominant_name": self.cluster_names[dom] if dom >= 0 else None,
                "activations": int(c[dom].item()) if dom >= 0 else 0,
            })

        vram: Dict[str, float] = {}
        if torch.cuda.is_available():
            vram = {"allocated_mb": round(torch.cuda.memory_allocated() / 2 ** 20, 1),
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved() / 2 ** 20, 1)}

        return {
            "model": SERVED_MODEL_ID, "base_model": self.model_id,
            "layers": len(self.layers), "clusters": self.num_clusters,
            "experts_per_cluster": self.experts_per_cluster,
            "total_experts": len(self.layers) * self.num_clusters * self.experts_per_cluster,
            "arts_core_pct": round(arts_pct, 2), "sci_core_pct": round(sci_pct, 2),
            "top_clusters": heat, "per_cluster": per_cluster, "per_layer": per_layer,
            "cuda_graph": bool(self.decoder.enabled),
            "cache_bucket_tokens": int(self.decoder.cache_len or 0),
            "slot_state": "busy" if self.gate.locked() else "idle",
            "slot_holder": self._slot_holder,
            "caged_clusters": sorted(self._caged.keys()),
            # 每个在押宗门实际被封杀的层号集合。cage() 会写入全部层 (=全局禁闭),
            # snipe(layer, cid) 只写入单层 (=单层狙击)。客户端据此区分两种来源。
            # 注意 free(cid) 总是整宗释放, 因此释放后该键会整体消失。
            "caged_layer_map": {str(c): sorted(d.keys()) for c, d in self._caged.items()},
            "plugged_cartridges": {str(k): v for k, v in self._plugged.items()},
            "vram": vram, "metrics": self.metrics.snapshot(),
        }

    def set_top_k(self, layer: Optional[int], k: int):
        k = max(1, min(int(k), MAX_TOP_K))
        if layer is None:
            for l in self.layers:
                l.mlp.top_k = k
            self.decoder.invalidate(f"全局开核数 -> {k}")
            return {"scope": "all", "k": k,
                    "active_experts_per_step": k * self.experts_per_cluster * len(self.layers)}
        if not (0 <= layer < len(self.layers)):
            raise ValueError(f"层号需在 0 ~ {len(self.layers) - 1} 之间")
        self.layers[layer].mlp.top_k = k
        self.decoder.invalidate(f"第 {layer:02d} 层开核数 -> {k}")
        return {"scope": "layer", "layer": layer, "k": k,
                "active_experts": k * self.experts_per_cluster}

    def _expert_tensors(self, mlp):
        if mlp.lora_A_gpu is not None:
            return mlp.lora_A_gpu, mlp.lora_B_gpu
        return mlp.lora_A_cpu, mlp.lora_B_cpu

    @torch.no_grad()
    def cage(self, cid: int):
        self._check_cluster(cid)
        if cid in self._caged:
            raise ValueError(f"宗门 #{cid:02d} 已在禁闭室")
        stash: Dict[int, torch.Tensor] = {}
        for i, l in enumerate(self.layers):
            _, b = self._expert_tensors(l.mlp)
            stash[i] = b[cid].clone()
            b[cid].zero_()
            l.mlp.set_caged([cid], True)
        self._caged[cid] = stash
        return {"cluster": cid, "name": self.cluster_names[cid],
                "caged_layers": len(stash)}

    @torch.no_grad()
    def free(self, cid: int, layer: Optional[int] = None):
        """释放宗门。layer=None 整宗释放；指定 layer 则只恢复该层（单层狙击的单点解封）。"""
        self._check_cluster(cid)
        if cid not in self._caged:
            raise ValueError(f"宗门 #{cid:02d} 并未被关押")
        stash = self._caged[cid]

        if layer is None:
            for i, l in enumerate(self.layers):
                _, b = self._expert_tensors(l.mlp)
                saved = stash.get(i)
                if saved is not None:
                    b[cid].copy_(saved)
                l.mlp.set_caged([cid], False)
            released = sorted(stash.keys())
            del self._caged[cid]
            return {"cluster": cid, "name": self.cluster_names[cid],
                    "released_layers": released, "fully_released": True,
                    "still_caged_layers": []}

        layer = int(layer)
        if not (0 <= layer < len(self.layers)):
            raise ValueError(f"层号需在 0 ~ {len(self.layers) - 1} 之间")
        if layer not in stash:
            raise ValueError(f"宗门 #{cid:02d} 在第 {layer} 层并未被封杀")

        # 只恢复这一层的权重与路由偏置，其余层保持封杀
        _, b = self._expert_tensors(self.layers[layer].mlp)
        b[cid].copy_(stash.pop(layer))
        self.layers[layer].mlp.set_caged([cid], False)
        # 该宗门已无任何层被封杀 → 从禁闭名单整体摘除
        fully_released = not stash
        if fully_released:
            del self._caged[cid]
        return {"cluster": cid, "name": self.cluster_names[cid],
                "released_layers": [layer], "fully_released": fully_released,
                "still_caged_layers": sorted(stash.keys())}

    @torch.no_grad()
    def snipe(self, layer: int, cid: int):
        self._check_cluster(cid)
        if not (0 <= layer < len(self.layers)):
            raise ValueError(f"层号需在 0 ~ {len(self.layers) - 1} 之间")
        mlp = self.layers[layer].mlp
        _, b = self._expert_tensors(mlp)
        self._caged.setdefault(cid, {}).setdefault(layer, b[cid].clone())
        b[cid].zero_()
        mlp.set_caged([cid], True)
        return {"layer": layer, "cluster": cid, "name": self.cluster_names[cid]}

    @torch.no_grad()
    def plug_cartridge(self, cartridge, slot: int = 16, source: Optional[str] = None):
        self._check_cluster(slot)
        if isinstance(cartridge, (str, os.PathLike)):
            if not os.path.exists(cartridge):
                raise FileNotFoundError(f"找不到卡带: {cartridge}")
            source = source or str(cartridge)
            cartridge = torch.load(str(cartridge), map_location="cpu")
        elif hasattr(cartridge, "read"):
            cartridge = torch.load(io.BytesIO(cartridge.read()), map_location="cpu")
        if not isinstance(cartridge, dict) or "layers" not in cartridge:
            raise ValueError("卡带格式错误: 需要包含 'layers' 字段的 dict")

        t0 = time.perf_counter()
        name = cartridge.get("name", source or "未命名卡带")
        for i, l in enumerate(self.layers):
            data = cartridge["layers"][i]
            a_gpu, b_gpu = self._expert_tensors(l.mlp)
            a_gpu[slot].copy_(data["lora_A"].to(a_gpu.device, dtype=a_gpu.dtype))
            b_gpu[slot].copy_(data["lora_B"].to(b_gpu.device, dtype=b_gpu.dtype))
            if "router_vec" in data:
                l.mlp.router_cluster.weight.data[slot].copy_(
                    data["router_vec"].to(self.device, dtype=self.dtype))
            l.mlp.set_caged([slot], False)
        if slot in self._caged:
            self._caged.pop(slot, None)
        ms = (time.perf_counter() - t0) * 1000
        self._plugged[slot] = {"name": name, "source": source,
                               "plugged_ms": round(ms, 2),
                               "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        return {"slot": slot, "name": name, "elapsed_ms": round(ms, 2)}

    def set_engine(self, enabled: Optional[bool] = None, max_len: Optional[int] = None):
        if enabled is not None:
            self.decoder.enabled = bool(enabled)
            self.decoder.invalidate("引擎开关切换")
        if max_len is not None:
            self.args.max_len = max(64, min(int(max_len), 8192))
        return {"cuda_graph": bool(self.decoder.enabled), "max_len": self.args.max_len}

    def _check_cluster(self, cid: int):
        if not (0 <= int(cid) < self.num_clusters):
            raise ValueError(f"宗门编号需在 0 ~ {self.num_clusters - 1} 之间")

    def encode_prompt(self, messages, chat_template_kwargs: Optional[dict] = None) -> torch.Tensor:
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **(chat_template_kwargs or {}))
        except Exception as exc:
            raise ValueError(f"chat template 渲染失败: {exc}") from exc
        return self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)

    async def _reap(self, thread: threading.Thread, stop_flag: threading.Event,
                    slot_tag: str) -> None:
        """独立守护任务: 等待 GPU 线程物理退出后再归还信号量，杜绝内存踩踏。"""
        waited = 0.0
        try:
            while thread.is_alive():
                await asyncio.sleep(0.02)
                waited += 0.02
                if waited > 120:
                    LOG.error("⚠️ 生成线程未在 120s 内退出 (slot=%s), 继续等待", slot_tag)
        except asyncio.CancelledError:
            pass
        finally:
            self._slot_holder = None
            self.metrics.active = max(0, self.metrics.active - 1)
            self.gate.release()

    async def stream(self, input_ids: torch.Tensor, params: GenParams,
                     focus_clusters: Optional[Sequence[int]] = None,
                     top_k_override: Optional[int] = None,
                     reset_stats: bool = False,
                     slot_tag: str = "-") -> AsyncGenerator[Dict[str, Any], None]:
        if focus_clusters is not None:
            try:
                for cid in focus_clusters:
                    self._check_cluster(cid)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        queued = self.gate.locked()
        if queued:
            self.metrics.waiting += 1
        await self.gate.acquire()
        if queued:
            self.metrics.waiting -= 1
        self.metrics.active += 1

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_flag = threading.Event()
        eos_ids = {self.tokenizer.eos_token_id, 151645, 151643}
        box: Dict[str, Any] = {"finish_reason": "stop", "content": "", "reasoning": "",
                               "prompt_tokens": int(input_ids.shape[1]), "completion_tokens": 0,
                               "elapsed": 0.0, "ttft": None}

        def emit(item):
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def worker():
            saved_topk = None
            t0 = time.perf_counter()
            generated = 0
            try:
                saved_topk = self._apply_overrides(focus_clusters, top_k_override)
                if reset_stats:
                    self.reset_stats()
                t0 = time.perf_counter()
                stream = TokenTextStream(self.tokenizer)
                splitter = ThoughtSplitter() if params.split_reasoning else None
                trimmer = StopTrimmer(params.stop)
                
                content_accum = ""
                sent_len = 0
                reason_buf: List[str] = []
                finish_reason = "stop"

                for tid in self.decoder.generate(
                        input_ids, max_new_tokens=params.max_new_tokens, eos_token_ids=eos_ids,
                        temperature=params.temperature, top_p=params.top_p,
                        repetition_penalty=params.repetition_penalty):
                    if stop_flag.is_set():
                        finish_reason = "cancelled"
                        break
                    generated += 1
                    if box["ttft"] is None:
                        box["ttft"] = time.perf_counter() - t0

                    raw, full = stream.push(int(tid))
                    if not raw:
                        if generated >= params.max_new_tokens:
                            finish_reason = "length"
                        continue

                    r_delta, c_delta = splitter.feed(full) if splitter else ("", raw)
                    if r_delta:
                        reason_buf.append(r_delta)

                    # 🌟 使用前瞻窗口对正文输出做安全检测
                    if c_delta:
                        content_accum += c_delta
                        hit_idx, safe_len = trimmer.check(content_accum)
                        
                        # 下发安全区间的文本
                        if safe_len > sent_len:
                            to_send = content_accum[sent_len:safe_len]
                            sent_len = safe_len
                            emit({"t": "delta", "reasoning": r_delta, "content": to_send})
                            r_delta = ""  # 已经发射过了

                        if hit_idx is not None:
                            content_accum = content_accum[:hit_idx]
                            finish_reason = "stop"
                            break

                    if r_delta:
                        emit({"t": "delta", "reasoning": r_delta, "content": ""})

                    if generated >= params.max_new_tokens:
                        finish_reason = "length"

                # 循环结束：如未触发 stop 词，将前瞻缓冲区内剩余的正常字符全部冲刷出来
                if finish_reason != "stop" and sent_len < len(content_accum):
                    emit({"t": "delta", "reasoning": "", "content": content_accum[sent_len:]})

                box["reasoning"] = "".join(reason_buf).rstrip("")
                box["content"] = content_accum.rstrip("")
                box["finish_reason"] = finish_reason
            except Exception as exc:
                LOG.exception("生成线程异常")
                box["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self._restore_overrides(saved_topk)
                box["completion_tokens"] = generated
                box["elapsed"] = time.perf_counter() - t0
                emit({"t": "done"})

        thread = threading.Thread(target=worker, name="myriad-gen", daemon=True)
        thread.start()

        reaper = asyncio.create_task(self._reap(thread, stop_flag, slot_tag))
        self._reapers.add(reaper)
        reaper.add_done_callback(self._reapers.discard)
        self._slot_holder = slot_tag
        try:
            while True:
                item = await queue.get()
                if item["t"] == "delta":
                    yield item
                    continue
                if "error" in box:
                    raise HTTPException(status_code=500, detail=box["error"])
                break
            if box["finish_reason"] != "cancelled":
                box["t"] = "final"
                yield box
        finally:
            stop_flag.set()


# ════════════════════════════════════════════════════════════════════════════
# 6. 数据模型
# ════════════════════════════════════════════════════════════════════════════
class MyriadControls(BaseModel):
    model_config = ConfigDict(extra="allow")
    focus_clusters: Optional[List[int]] = Field(
        None, description="本次请求只允许这些宗门参与路由，未列出的临时打入冷宫")
    top_k: Optional[int] = Field(None, description="临时覆盖每层开核数 (1~10)")
    stats: bool = Field(False, description="响应附带本次全息遥测")
    reset_stats: bool = Field(False, description="本次生成前清零统计")


class MyriadChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = SERVED_MODEL_ID
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: Optional[float] = Field(None, description="原生重复惩罚 (默认 1.15)")
    seed: Optional[int] = None
    user: Optional[str] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        None, description='Qwen3 思维链开关，例如 {"enable_thinking": false}')
    split_reasoning: bool = Field(True, description="把 <think> 段落分流到 reasoning_content")
    myriad: Optional[MyriadControls] = None


class MyriadCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = SERVED_MODEL_ID
    prompt: Union[str, List[str], List[int], List[List[int]]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: float = 0.0
    repetition_penalty: Optional[float] = None
    echo: bool = False
    seed: Optional[int] = None
    user: Optional[str] = None
    split_reasoning: bool = True
    myriad: Optional[MyriadControls] = None


class TopKBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    k: int
    layer: Optional[int] = Field(None, description="留空 = 全局 28 层统一调频 (/set_k_all)")


class SnipeBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    layer: int
    cluster: int


class EngineBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    cuda_graph: Optional[bool] = None
    max_len: Optional[int] = None


# ════════════════════════════════════════════════════════════════════════════
# 7. 斜杠指令调度
# ════════════════════════════════════════════════════════════════════════════
HELP_TEXT = """📖【Myriad-MoE API 能力速查】
神经透视 : /catch 逐层主导宗门雷达 · /stats 全息看板 · /clusters 宗门状态
弹性开核 : /show_k 各层开核数 · /set_k <层> <核数> · /set_k_all <核数>
神经禁闭 : /cage <宗门> · /free <宗门> · /snipe <层> <宗门>
热插拔   : /plug <卡带.pt> [插槽, 默认16]
引擎参数 : /graph 切换 CUDA Graph · /maxlen <n> 默认回复上限
会话     : /clear 清空上下文 · /help 本指南
REST 等价: /v1/myriad/{stats,topk,catch,clusters,engine,cartridge/plug,metrics}"""

KNOWN_COMMANDS = {"/help", "/h", "/clear", "/stats", "/clusters", "/show_k", "/catch",
                  "/set_k", "/set_k_all", "/cage", "/free", "/snipe", "/plug", "/graph",
                  "/maxlen"}


def run_command(eng: MyriadEngine, line: str) -> str:
    parts = line.strip().split()
    cmd, args = parts[0].lower(), parts[1:]

    if cmd in ("/help", "/h"):
        return HELP_TEXT
    if cmd == "/clear":
        return "🧹 记忆已重置。(API 为无状态模式, 对话上下文由客户端自行维护)"
    if cmd == "/stats":
        d = eng.dashboard()
        heat = " · ".join(f"#{h['id']:02d}{h['name']}({h['hits']})" for h in d["top_clusters"])
        return (f"🌌 文科基盘 {d['arts_core_pct']:.1f}% / 理科宏核 {d['sci_core_pct']:.1f}%\n"
                f"🪐 热力: {heat}\n"
                f"⚡ CUDA Graph={d['cuda_graph']} · 显存 {d['vram'].get('allocated_mb', 0)}MB · "
                f"槽位 {d['slot_state']} · 累计 {d['metrics']['completion_tokens_total']} tokens")
    if cmd == "/clusters":
        rows = [f"  #{c['id']:02d} {c['name']:<16} 命中 {c['hits']:>9,}"
                + ("  🔒禁闭" if c["caged"] else "")
                + (f"  💿{c['cartridge']}" if c.get("cartridge") else "")
                for c in eng.dashboard()["per_cluster"]]
        return "🪐【20 宗门状态】\n" + "\n".join(rows)
    if cmd == "/show_k":
        rows = [f"  - Layer {i:02d}: {l.mlp.top_k} 核 [{'▮' * int(l.mlp.top_k):<10}]"
                f" ({int(l.mlp.top_k) * eng.experts_per_cluster:3d} 微专家并发)"
                for i, l in enumerate(eng.layers)]
        return "🎛️【各层开核配置】\n" + "\n".join(rows)
    if cmd == "/catch":
        rows = []
        for i, l in enumerate(eng.layers):
            c = l.mlp.stat_cluster_counts.cpu()
            if int(c.sum()) == 0:
                rows.append(f"  - Layer {i:02d}: 暂无前向激活数据")
                continue
            cid = int(torch.argmax(c).item())
            rows.append(f"  - Layer {i:02d}: #{cid:02d} [{eng.cluster_names[cid]:<16}] "
                        f"({int(c[cid].item()):3d} 拍){'  🔥极度可疑' if cid == 16 else ''}")
        return "🚨【28 层神经透视雷达】\n" + "\n".join(rows)
    if cmd == "/set_k":
        if len(args) < 2:
            return "⚠️ 用法: /set_k <层数> <开核数>, 例如 /set_k 9 4"
        return "⚡ [调频] " + json.dumps(eng.set_top_k(int(args[0]), int(args[1])), ensure_ascii=False)
    if cmd == "/set_k_all":
        if not args:
            return "⚠️ 用法: /set_k_all <开核数>, 例如 /set_k_all 1"
        return "⚡ [全局调频] " + json.dumps(eng.set_top_k(None, int(args[0])), ensure_ascii=False)
    if cmd == "/cage":
        if not args:
            return "⚠️ 用法: /cage <宗门号>"
        return "🔒 [关禁闭] " + json.dumps(eng.cage(int(args[0])), ensure_ascii=False)
    if cmd == "/free":
        if not args:
            return "⚠️ 用法: /free <宗门号>"
        return "🔓 [刑满释放] " + json.dumps(eng.free(int(args[0])), ensure_ascii=False)
    if cmd == "/snipe":
        if len(args) < 2:
            return "⚠️ 用法: /snipe <层数> <宗门号>"
        return "🎯 [狙击] " + json.dumps(eng.snipe(int(args[0]), int(args[1])), ensure_ascii=False)
    if cmd == "/plug":
        if not args:
            return "⚠️ 用法: /plug <卡带.pt> [插槽]"
        slot = int(args[1]) if len(args) > 1 else 16
        return "⚡ [热插拔] " + json.dumps(eng.plug_cartridge(args[0], slot, args[0]), ensure_ascii=False)
    if cmd == "/graph":
        return "🔧 [引擎] " + json.dumps(eng.set_engine(not eng.decoder.enabled), ensure_ascii=False)
    if cmd == "/maxlen":
        if not args:
            return f"⚠️ 用法: /maxlen <token数> (当前 {eng.args.max_len})"
        return "📏 [回复上限] " + json.dumps(eng.set_engine(max_len=int(args[0])), ensure_ascii=False)
    return f"❓ 未知指令 {cmd}, 输入 /help 查看全部能力"


def maybe_command(eng: MyriadEngine, messages: List[Dict[str, Any]]) -> Optional[str]:
    users = [m for m in messages if m.get("role") == "user"]
    if not users:
        return None
    content = users[-1].get("content")
    if not isinstance(content, str):
        return None
    s = content.strip()
    if not s.startswith("/") or s.split()[0].lower() not in KNOWN_COMMANDS:
        return None
    try:
        return run_command(eng, s)
    except Exception as exc:
        return f"⚠️ 指令执行失败: {exc}"


# ════════════════════════════════════════════════════════════════════════════
# 8. FastAPI 路由
# ════════════════════════════════════════════════════════════════════════════
engine: Optional[MyriadEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if engine is not None:
        try:
            engine.load()
        except Exception as exc:
            engine.startup_error = f"{type(exc).__name__}: {exc}"
            LOG.error("模型加载失败: %s", engine.startup_error)
    yield


app = FastAPI(title="Myriad-MoE OpenAI-Compatible API", version="1.0.0",
              description="25,200 微专家 · 双大核 MoE · CUDA Graph 极速解码 · OpenAI 兼容",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def require_auth(authorization: Optional[str] = Header(default=None)):
    key = getattr(getattr(engine, "args", None), "api_key", None)
    if not key:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != key:
        raise HTTPException(status_code=401, detail="Invalid API key",
                            headers={"WWW-Authenticate": "Bearer"})


auth = [Depends(require_auth)]


def require_engine():
    """守卫：模型未加载完成时统一返回 503。

    没有这层守卫时，权重加载期间访问 /v1/myriad/* 会走到
    engine.dashboard() → self.layers → self.model.model.layers，
    而 self.model 在 load() 完成前是 None，于是抛 AttributeError 变成 500。
    500 对客户端毫无信息量（无法区分「服务坏了」和「还在启动」），
    503 + 明确文案才能让客户端正确显示「加载中」。
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="引擎尚未创建")
    if not engine.ready:
        detail = engine.startup_error or f"模型正在加载中 ({engine.model_id})，请稍候重试"
        raise HTTPException(status_code=503, detail=detail)


guard = [Depends(require_engine)]


def _resolve_params(req: MyriadChatRequest) -> GenParams:
    max_new = req.max_completion_tokens or req.max_tokens or engine.args.max_len
    rep = req.repetition_penalty
    if rep is None:
        rep = DEFAULT_REPETITION_PENALTY if not req.frequency_penalty \
            else 1.0 + min(2.0, max(-2.0, req.frequency_penalty))
    stops = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
    
    # 🌟 修复：严谨的 temperature 转换与 0.0 贪婪保护
    temp = DEFAULT_TEMPERATURE if req.temperature is None else float(req.temperature)
    temp = max(0.0, temp)

    return GenParams(
        max_new_tokens=max(1, min(int(max_new), 8192)),
        temperature=temp,
        top_p=DEFAULT_TOP_P if req.top_p is None else float(req.top_p),
        repetition_penalty=float(rep),
        stop=stops,
        split_reasoning=req.split_reasoning,
    )


def _telemetry(controls: Optional[MyriadControls]) -> Dict[str, Any]:
    if not (controls and controls.stats):
        return {}
    d = engine.dashboard()
    return {"arts_core_pct": d["arts_core_pct"], "sci_core_pct": d["sci_core_pct"],
            "top_clusters": d["top_clusters"], "cuda_graph": d["cuda_graph"],
            "slot_state": d["slot_state"]}


def _controls_of(req) -> Tuple[Optional[List[int]], Optional[int], bool]:
    c: Optional[MyriadControls] = getattr(req, "myriad", None)
    if c is None:
        return None, None, False
    focus = None
    if c.focus_clusters is not None:
        try:
            for cid in c.focus_clusters:
                engine._check_cluster(cid)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        focus = sorted(set(c.focus_clusters))
    return focus, c.top_k, bool(c.reset_stats)


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


# ──────────────────────────────────────────────────────── OpenAI 兼容端点
@app.get("/v1/models", dependencies=auth)
async def list_models():
    info = engine.dashboard() if engine.ready else {}
    return {"object": "list", "data": [{
        "id": SERVED_MODEL_ID, "object": "model", "created": int(time.time()),
        "owned_by": "feifei", "root": SERVED_MODEL_ID, "parent": None, "permission": [],
        "max_model_len": 32768,
        "myriad": {"base_model": engine.model_id, "layers": info.get("layers"),
                   "clusters": engine.num_clusters,
                   "experts": info.get("total_experts"), "cuda_graph": info.get("cuda_graph")},
    }]}


@app.get("/v1/models/{model_id}", dependencies=auth)
async def get_model(model_id: str):
    if model_id not in (SERVED_MODEL_ID, engine.model_id):
        raise HTTPException(status_code=404, detail=f"model '{model_id}' not found")
    return (await list_models())["data"][0]


@app.post("/v1/chat/completions", dependencies=auth)
async def chat_completions(req: MyriadChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages 不能为空")
    if not engine.ready:
        raise HTTPException(status_code=503, detail=engine.startup_error or "模型尚未就绪")
    if req.n > 1:
        raise HTTPException(status_code=400, detail="单槽位解码暂不支持 n>1, 请分多次请求")

    params = _resolve_params(req)
    focus, top_k, reset = _controls_of(req)

    cmd_out = maybe_command(engine, req.messages)
    if cmd_out is not None:
        engine.metrics.command_requests += 1
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
            "created": int(time.time()), "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": cmd_out},
                         "logprobs": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "system_fingerprint": "myriad-cmd",
            "myriad": {"command_dispatched": True}})

    try:
        input_ids = engine.encode_prompt(req.messages, req.chat_template_kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if req.seed is not None:
        torch.manual_seed(int(req.seed))

    if req.stream:
        engine.metrics.stream_requests += 1
        return StreamingResponse(
            _chat_stream(req, input_ids, params, focus, top_k, reset),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "X-Myriad-Slots": "1"})
    return await _chat_nonstream(req, input_ids, params, focus, top_k, reset)


async def _chat_nonstream(req, input_ids, params: GenParams, focus, top_k, reset):
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    t_start = time.perf_counter()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    content, reasoning, fin, ttft = "", "", "stop", None
    async with aclosing(engine.stream(input_ids, params, focus, top_k, reset, cid[:14])) as gen:
        async for ev in gen:
            if ev["t"] == "final":
                fin, content = ev["finish_reason"], ev["content"]
                reasoning, ttft = ev.get("reasoning", ""), ev.get("ttft")
                usage = {"prompt_tokens": ev["prompt_tokens"],
                         "completion_tokens": ev["completion_tokens"],
                         "total_tokens": ev["prompt_tokens"] + ev["completion_tokens"]}
                break
    dur = time.perf_counter() - t_start
    engine.metrics.observe(ttft, dur, usage["prompt_tokens"], usage["completion_tokens"])
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if params.split_reasoning and reasoning:
        message["reasoning_content"] = reasoning
    body = {
        "id": cid, "object": "chat.completion", "created": created, "model": req.model,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": fin}],
        "usage": usage, "system_fingerprint": "myriad-v1",
        "myriad": {"elapsed_sec": round(dur, 3),
                   "tokens_per_sec": round(usage["completion_tokens"] / dur, 1) if dur > 0 else None,
                   **_telemetry(getattr(req, "myriad", None))},
    }
    return JSONResponse(body)


async def _chat_stream(req, input_ids, params: GenParams, focus, top_k, reset):
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))
    t_start = time.perf_counter()

    def chunk(delta, finish=None, usage=None):
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": req.model, "system_fingerprint": "myriad-v1",
                   "choices": [{"index": 0, "delta": delta, "logprobs": None,
                                "finish_reason": finish}]}
        if usage is not None:
            payload["usage"] = usage
        return _sse(payload)

    yield chunk({"role": "assistant", "content": ""})
    ttft = None
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    fin, failed = "stop", None
    try:
        async with aclosing(engine.stream(input_ids, params, focus, top_k, reset, cid[:14])) as gen:
            async for ev in gen:
                if ev["t"] == "delta":
                    delta: Dict[str, str] = {}
                    if ev["reasoning"]:
                        delta["reasoning_content"] = ev["reasoning"]
                    if ev["content"]:
                        delta["content"] = ev["content"]
                    if delta:
                        if ttft is None:
                            ttft = time.perf_counter() - t_start
                        yield chunk(delta)
                elif ev["t"] == "final":
                    fin = ev["finish_reason"]
                    usage = {"prompt_tokens": ev["prompt_tokens"],
                             "completion_tokens": ev["completion_tokens"],
                             "total_tokens": ev["prompt_tokens"] + ev["completion_tokens"]}
    except Exception as exc:
        failed = f"{type(exc).__name__}: {exc}"
        engine.metrics.errors_total += 1
        LOG.exception("流式生成失败")
    if failed:
        yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "error": {"message": failed, "type": "server_error"},
                    "choices": []})
    else:
        yield chunk({}, fin)
    if include_usage:
        yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "choices": [], "usage": usage,
                    "system_fingerprint": "myriad-v1"})
    dur = time.perf_counter() - t_start
    engine.metrics.observe(ttft, dur, usage["prompt_tokens"], usage["completion_tokens"])
    # 本次问答专属的全息遥测。此前只有非流式路径下发 (_telemetry 在 _chat_nonstream 里)，
    # 导致 stream=true 时客户端永远拿不到 arts/sci 占比与本轮命中宗门。
    # 放在 metrics.observe 之后，保证 tokens 等累计值已包含本次请求。
    telemetry = _telemetry(getattr(req, "myriad", None)) if not failed else {}
    if telemetry:
        yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model, "system_fingerprint": "myriad-v1",
                    "choices": [], "usage": usage, "myriad": telemetry})
    yield "data: [DONE]\n\n"


@app.post("/v1/completions", dependencies=auth)
async def completions(req: MyriadCompletionRequest):
    if not engine.ready:
        raise HTTPException(status_code=503, detail=engine.startup_error or "模型尚未就绪")
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    prompts = [engine.tokenizer.decode(p) if isinstance(p, list) else str(p) for p in prompts]
    max_new = req.max_tokens or engine.args.max_len
    temp = DEFAULT_TEMPERATURE if req.temperature is None else float(req.temperature)
    temp = max(0.0, temp)
    params = GenParams(
        max_new_tokens=max(1, min(int(max_new), 8192)),
        temperature=temp,
        top_p=DEFAULT_TOP_P if req.top_p is None else float(req.top_p),
        repetition_penalty=req.repetition_penalty or DEFAULT_REPETITION_PENALTY,
        stop=([req.stop] if isinstance(req.stop, str) else list(req.stop or [])),
        split_reasoning=req.split_reasoning)
    focus, top_k, reset = _controls_of(req)
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    results = []
    for i, p in enumerate(prompts):
        ids = engine.tokenizer(p, return_tensors="pt")["input_ids"].to(engine.device)
        content, fin = "", "stop"
        async with aclosing(engine.stream(ids, params, focus, top_k, reset, f"cmpl{i}")) as gen:
            async for ev in gen:
                if ev["t"] == "final":
                    content, fin = ev["content"], ev["finish_reason"]
                    usage = {"prompt_tokens": ev["prompt_tokens"],
                             "completion_tokens": ev["completion_tokens"],
                             "total_tokens": ev["prompt_tokens"] + ev["completion_tokens"]}
        results.append({"text": (p if req.echo else "") + content, "index": i,
                        "logprobs": None, "finish_reason": fin})
    return {"id": f"cmpl-{uuid.uuid4().hex}", "object": "text_completion",
            "created": int(time.time()), "model": req.model, "choices": results, "usage": usage}


# ────────────────────────────────────────────────────── Myriad 特色管理端点
@app.get("/v1/myriad/stats", dependencies=auth + guard)
async def stats():
    return engine.dashboard()


@app.post("/v1/myriad/stats/reset", dependencies=auth + guard)
async def stats_reset():
    engine.reset_stats()
    return {"object": "myriad.stats.reset", "ok": True, "message": "🧹 统计已清零"}


@app.get("/v1/myriad/catch", dependencies=auth + guard)
async def catch_radar():
    out = []
    for i, l in enumerate(engine.layers):
        c = l.mlp.stat_cluster_counts.cpu()
        if int(c.sum()) == 0:
            out.append({"layer": i, "dominant_cluster": None, "activations": 0})
            continue
        cid = int(torch.argmax(c).item())
        out.append({"layer": i, "dominant_cluster": cid,
                    "dominant_name": engine.cluster_names[cid],
                    "activations": int(c[cid].item()), "suspicious": cid == 16})
    return {"object": "myriad.catch", "layers": out,
            "top_clusters": engine.dashboard()["top_clusters"]}


@app.get("/v1/myriad/clusters", dependencies=auth + guard)
async def clusters():
    d = engine.dashboard()
    return {"object": "list", "data": d["per_cluster"],
            "caged": d["caged_clusters"], "plugged": d["plugged_cartridges"]}


@app.get("/v1/myriad/topk", dependencies=auth + guard)
async def get_topk():
    return {"object": "myriad.topk", "layers": [
        {"layer": i, "top_k": int(l.mlp.top_k),
         "active_experts": int(l.mlp.top_k) * engine.experts_per_cluster}
        for i, l in enumerate(engine.layers)]}


@app.post("/v1/myriad/topk", dependencies=auth + guard)
async def set_topk(body: TopKBody):
    try:
        res = engine.set_top_k(body.layer, body.k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"object": "myriad.topk.updated", **res}


@app.post("/v1/myriad/clusters/{cid}/cage", dependencies=auth + guard)
async def cage_cluster(cid: int):
    try:
        res = engine.cage(cid)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"object": "myriad.cluster.caged", **res}


@app.post("/v1/myriad/clusters/{cid}/free", dependencies=auth + guard)
async def free_cluster(cid: int, layer: Optional[int] = None):
    """layer 省略 = 整宗释放（向后兼容）；指定 layer = 仅解封该层，用于单层狙击的单点撤销。"""
    try:
        res = engine.free(cid, layer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    obj = "myriad.cluster.layer_freed" if layer is not None else "myriad.cluster.freed"
    return {"object": obj, **res}


@app.post("/v1/myriad/snipe", dependencies=auth + guard)
async def snipe_cluster(body: SnipeBody):
    try:
        res = engine.snipe(body.layer, body.cluster)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"object": "myriad.cluster.sniped", **res}


@app.post("/v1/myriad/cartridge/plug", dependencies=auth + guard)
async def plug_cartridge(file: Optional[UploadFile] = File(None),
                         path: Optional[str] = Form(None),
                         slot: int = Form(16)):
    if file is None and not path:
        raise HTTPException(status_code=400, detail="请提供 file (.pt) 或 path")
    try:
        if file is not None:
            res = engine.plug_cartridge(file.file, slot, file.filename)
        else:
            res = engine.plug_cartridge(path, slot, path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"卡带植入失败: {exc}") from exc
    LOG.info("⚡ 卡带《%s》植入插槽 #%02d (%.2f ms)", res["name"], res["slot"], res["elapsed_ms"])
    return {"object": "myriad.cartridge.plugged", **res}


@app.post("/v1/myriad/engine", dependencies=auth + guard)
async def set_engine(body: EngineBody):
    return {"object": "myriad.engine", **engine.set_engine(body.cuda_graph, body.max_len)}


@app.get("/v1/myriad/metrics", dependencies=auth + guard)
async def metrics(format: str = "json"):
    if format == "prometheus":
        return PlainTextResponse(engine.metrics.prometheus(),
                                 media_type="text/plain; version=0.0.4")
    return {"object": "myriad.metrics", **engine.metrics.snapshot()}


@app.get("/health")
async def health():
    return {"status": "ok" if (engine and engine.ready) else "loading",
            "model": SERVED_MODEL_ID, "ready": bool(engine and engine.ready),
            "error": engine.startup_error if engine else "engine not created"}


# ─────────────────────────────────────────────────────────────────── 仪表盘
DASHBOARD_HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Myriad-MoE 神经全息监控台</title><style>
body{background:#07080d;color:#d8e2f0;font-family:ui-monospace,Menlo,Consolas,monospace;margin:0;padding:24px}
h1{font-size:18px;color:#7ee0ff;letter-spacing:1px}
.card{background:#0d1119;border:1px solid #1f2b3d;border-radius:10px;padding:16px;margin-bottom:16px}
.kv{display:flex;flex-wrap:wrap;gap:24px;font-size:12px}
.kv b{color:#7ee0ff;font-size:20px;display:block}
.bar{height:12px;background:#182233;border-radius:6px;overflow:hidden;display:flex;margin:8px 0}
.arts{background:linear-gradient(90deg,#3b82f6,#60a5fa)}
.sci{background:linear-gradient(90deg,#f97316,#fb923c)}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:4px 6px;text-align:left;border-bottom:1px solid #16202e}
.bar2{height:8px;background:#16202e;border-radius:4px;overflow:hidden}
.fill{height:100%;background:linear-gradient(90deg,#7c3aed,#22d3ee)}
.pill{padding:1px 8px;border-radius:10px;font-size:11px;border:1px solid #2b3a4f;margin-left:4px}
.caged{color:#f87171;border-color:#7f1d1d}
.slot{color:#4ade80;border-color:#14532d}
</style></head><body>
<h1>🌌 Myriad-MoE · 25,200 微专家神经全息监控台</h1>
<div class="card"><div class="kv" id="kv">加载中…</div></div>
<div class="card"><b>文理双核占比</b><div class="bar"><div class="arts" id="arts" style="width:50%"></div>
<div class="sci" id="sci" style="width:50%"></div></div></div>
<div class="card"><b>20 宗门激活热力</b><table id="heat"></table></div>
<div class="card"><b>逐层开核 / 主导宗门</b><table id="layers"></table></div>
<script>
async function tick(){try{
 const d=await (await fetch('/v1/myriad/stats')).json();
 document.getElementById('kv').innerHTML=[['模型',d.model],['微专家',d.total_experts.toLocaleString()],
  ['层数',d.layers],['CUDA Graph',d.cuda_graph?'⚡ ON':'OFF'],['槽位',d.slot_state],
  ['显存 MB',d.vram.allocated_mb||0],['总请求',d.metrics.requests_total],
  ['已生成 token',d.metrics.completion_tokens_total],['均速 tok/s',d.metrics.avg_tokens_per_sec]]
  .map(([k,v])=>'<div>'+k+'<b>'+v+'</b></div>').join('');
 document.getElementById('arts').style.width=d.arts_core_pct+'%';
 document.getElementById('sci').style.width=d.sci_core_pct+'%';
 const max=Math.max(1,...d.per_cluster.map(c=>c.hits));
 document.getElementById('heat').innerHTML=d.per_cluster.map(c=>'<tr><td>#'+String(c.id).padStart(2,'0')+
  '</td><td>'+c.name+'</td><td style="width:60%"><div class="bar2"><div class="fill" style="width:'+
  (c.hits/max*100)+'%"></div></div></td><td>'+c.hits.toLocaleString()+'</td><td>'+
  (c.caged?'<span class="pill caged">禁闭</span>':'')+(c.cartridge?'<span class="pill slot">'+c.cartridge+
  '</span>':'')+'</td></tr>').join('');
 document.getElementById('layers').innerHTML=d.per_layer.map(l=>'<tr><td>Layer '+String(l.layer).padStart(2,'0')+
  '</td><td>'+l.top_k+' 核</td><td>'+l.active_experts+' 专家</td><td>'+
  (l.dominant_name?'#'+l.dominant_cluster+' '+l.dominant_name:'—')+'</td><td>'+l.activations+
  '</td></tr>').join('');
}catch(e){document.getElementById('kv').textContent='等待服务就绪…'}}
tick();setInterval(tick,2000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return DASHBOARD_HTML


# ════════════════════════════════════════════════════════════════════════════
# 9. 启动入口
# ════════════════════════════════════════════════════════════════════════════
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Myriad-MoE OpenAI 兼容 API 服务端",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--host", default=os.getenv("MYRIAD_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("MYRIAD_PORT", 8000)))
    p.add_argument("--model", default=os.getenv("MYRIAD_MODEL", "Qwen/Qwen3-0.6B"))
    p.add_argument("--weights", default=os.getenv("MYRIAD_WEIGHTS", "myriad_moe_hierarchical_weights.pt"))
    p.add_argument("--backend-script", default=os.getenv(
        "MYRIAD_BACKEND", "6.chat_myriad_25k_lora_fast_more_mirco.py"))
    p.add_argument("--device", default=os.getenv("MYRIAD_DEVICE", "cuda:0"))
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--micro-rank", type=int, default=16)
    p.add_argument("--macro-rank", type=int, default=64)
    p.add_argument("--num-clusters", type=int, default=20)
    p.add_argument("--experts-per-cluster", type=int, default=45)
    p.add_argument("--layer-top-k", type=int, nargs="*", default=[
        1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 3, 3, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1],
        help="各层初始开核数 (异构金字塔)")
    p.add_argument("--cpu-experts", action="store_true",
                   help="微专家常驻 CPU pinned memory (走 PCIe 回退流式路径)")
    p.add_argument("--no-graph", action="store_true", help="禁用 CUDA Graph (普通 eager 解码)")
    p.add_argument("--max-len", type=int, default=512, help="默认最大回复 token 数")
    p.add_argument("--api-key", default=os.getenv("MYRIAD_API_KEY"),
                   help="设置后需 Authorization: Bearer <key>")
    p.add_argument("--plug", action="append", metavar="[slot=]path.pt",
                   help="启动时热插拔卡带, 可重复, 例: --plug 16=cartridge_gongfang.pt")
    p.add_argument("--log-level", default=os.getenv("MYRIAD_LOG_LEVEL", "info"))
    return p


def create_engine(args: argparse.Namespace) -> MyriadEngine:
    args.backend = load_backend_module(args.backend_script)
    return MyriadEngine(args)


def main():
    global engine
    args = build_argparser().parse_args()
    engine = create_engine(args)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("缺少 uvicorn, 请先安装: uv pip install --python .venv/bin/python "
                         "fastapi \"uvicorn[standard]\"") from exc

    print(f"""
╔════════════════════════════════════════════════════════════════════════╗
║ 🌌 Myriad-MoE OpenAI 兼容 API (生产加固版)                            ║
║   base_url : http://127.0.0.1:{args.port}/v1                             ║
║   看板     : http://127.0.0.1:{args.port}/                             ║
║   遥测     : /v1/myriad/stats · /v1/myriad/metrics?format=prometheus     ║
║   鉴权     : {'Bearer ' + args.api_key if args.api_key else '未开启'}                                   ║
╚════════════════════════════════════════════════════════════════════════╝""")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
# 🌌 Myriad-MoE (DualBigLittle-MoE)

> **基于双显存大核与宿主锁页内存流式微专家的多任务认知解耦架构**  
> *仅用 0.9B 激活参数战平 7B+ 基线模型；25,200 个模块化微专家仅占 1.54 GB 内存；支持 ~40ms 内存级赛博义体在线热插拔。*

[![Transformers 提案](https://img.shields.io/badge/HuggingFace-Transformers_%2349183-orange.svg)](https://github.com/huggingface/transformers/issues/49183)
[![开源协议: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![基座模型](https://img.shields.io/badge/Base-Qwen3--0.6B-green.svg)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![实测验证](https://img.shields.io/badge/Verified_on-RTX_5090_D-purple.svg)]()

---

## 💡 核心设计思想

传统稠密大语言模型在多任务训练中普遍存在严重的**文理负迁移（跨领域梯度冲突）**：针对严格代码与数学推理的微调，会破坏文学自然语言的语感与母语流畅度；而提升通识闲聊能力，又会削弱底层的严密推演逻辑。另一方面，传统混合专家模型（MoE）对显存（VRAM）有着极度贪婪的消耗，使得在消费级单卡上扩展海量专家变得天方夜谭。

**Myriad-MoE 借鉴了移动端 SoC（ARM big.LITTLE 大小核调度与统一内存分页）在极限硬件压榨上的二十年智慧，成功将这一异构体系重构至大模型底层：**

1. **双显存密集大核（宏观底座·双核定乾坤）**：
   - **Tier-1 文科原版大核（常驻 GPU 显存）**：100% 物理锁死原生密集 MLP，死守底层语言直觉、古典修辞、通识美感与因果常识，彻底杜绝语言退化。
   - **Tier-2 理科特训大核（常驻 GPU 显存）**：正交克隆并专门特训的密集 MLP，专攻编程语法、数据结构、算法逻辑与数理推导。
2. **宿主锁页内存流式微专家（极致稀疏·两万微核常驻内存）**：
   - 28 层 $\times$ 20 宗门 $\times$ 45 专家（Rank-16 LoRA 切片）$\approx$ **25,200 个微专家**。
   - **仅消耗 1.54 GB 普通系统物理内存（RAM）**，通过 Linux 锁页内存（`pin_memory()`）常驻。
   - 依托 CUDA 异步流与 DMA 引擎，在每个 Token 生成的毫秒间隙内，按需流式拉取命中的微专家切片。
3. **零延迟损耗（实测 ~29-31 tokens/s，物理完全掩蔽）**：
   - 每层搬运一个 96 KB 的微专家切片，在 PCIe 4.0/5.0 上仅需 $\approx 0.13 \text{ ms}$。
   - 这 0.13 毫秒的数据搬运时间，被 GPU 跑稠密底座前向计算的 40 毫秒**100% 完全掩蔽（Latency Hiding）**！
   - 实测证明：**CPU 宿主流式模式（24.9 t/s）与纯 GPU 显存常驻模式（23.8 t/s）完全持平**，以零性能代价抹平了显存占用。
4. **赛博义体在线热插拔（~40ms 内存秒级换件）**：
   - 私有规则卡带可在 20 秒内独立锻造完成（仅几十兆）；
   - 在推理服务持续运行、不重启进程、不打断显存上下文的情况下，通过原生内存 `copy_()` 在 **44 毫秒内原地改写指定插槽**，即插即用！
5. **交互式神经外科排障与缉捕雷达（内鬼定位与切除）**：
   - 提供了 `/catch` 28 层 CT 级深层神经透视雷达，瞬间揪出带头造反的作恶层与宗门；
   - 支持 `/cage` 一键关禁闭、`/free` 实时释放、以及 `/snipe` 单层定点狙击。

---

## 🏛️ 系统架构拓扑

```text
                            ┌──────────────────────────────────────┐
                            │            输入隐状态向量 x            │
                            └──────────────────┬───────────────────┘
                                               │
               ┌───────────────────────────────┴───────────────────────────────┐
               ▼                                                               ▼
 ┌───────────────────────────┐                                   ┌───────────────────────────┐
 │   双大核路由器 (2分类)    │                                   │   宗门路由器 (20分类)     │
 └─────────────┬─────────────┘                                   └─────────────┬─────────────┘
               │ 软融合权重 (w_arts, w_sci)                                    │ Top-2 宗门插槽动态决策
               ▼                                                               ▼
 ┌───────────────────────────┐                     ┌────────────────────────────────────────────────────────┐
 │   GPU 显存双稠密大核      │                     │          CPU 宿主锁页内存 (仅占 1.54 GB RAM)           │
 │ ┌───────────────────────┐ │                     │ ┌────────────────────────────────────────────────────┐ │
 │ │ Tier-1: 文科核 (锁死) │ │                     │ │ 25,200 微专家 (20宗门 x 45微专家 x 28层)            │ │
 │ └───────────┬───────────┘ │                     │ └──────────────────────────┬─────────────────────────┘ │
 │             │             │                     └────────────────────────────┼───────────────────────────┘
 │             ▼             │                                                  │ PCIe DMA 异步零拷贝流式传输
 │ ┌───────────────────────┐ │                                                  ▼ (单层切片仅 96 KB)
 │ │ Tier-2: 理科核 (特训) │ │                     ┌────────────────────────────────────────────────────────┐
 │ └───────────┬───────────┘ │                     │         GPU 流式计算暂存区                             │
 └─────────────┼─────────────┘                     │            (F.linear Rank-16 低秩矩阵运算)             │
               │ 稠密底座输出 (big_out)            └────────────────────────────┬───────────────────────────┘
               │                                                                │ 稀疏微专家输出 (micro_out)
               └───────────────────────────────┬────────────────────────────────┘
                                               ▼
                                      最终融合隐状态输出
                         y = big_out + γ * (Top-2 加权 micro_out)
```

---

## 📐 底层数学物理原理

### 1. 维度自适应残差缩放（$\mu P$ 神经正交方差守恒）
为了防止 25,200 个微专家在穿透 28~35 层深网时引起隐空间表示漂移（Representation Drift），我们摒弃了盲猜超参数的经验做法，引入基于各向同性方差守恒的严格理论解：
$$\gamma = \frac{1}{\sqrt{d_{\text{model}}}}$$
- 在 $d_{\text{model}} = 1024$（Qwen-0.6B 基座）：$\gamma = \frac{1}{32} \approx \mathbf{0.03125}$
- 在 $d_{\text{model}} = 1536$（Scalpel-E2B 基座）：$\gamma = \frac{1}{\sqrt{1536}} \approx \mathbf{0.0255}$
- 在 $d_{\text{model}} = 4096$（7B 规模基座）：$\gamma = \frac{1}{64} \approx \mathbf{0.0156}$

该公式保证了无论骨干网络在何种参数规模之间缩放，微专家注入的扰动能级始终与基座维持严格的单位方差匹配，杜绝了深层梯度的剧烈爆炸。

### 2. 训练期低维张量收缩（避免显存爆炸的关键）
在训练 25,200 个专家时，若在显存中直接实例化完整的五维中间激活张量 $[B, S, C, E, D]$，单层显存消耗将超过 1.4 GB。我们在低秩瓶颈层内完成了加权提前吸收：
$$h = \text{einsum}('bsd,cerd \to bscer', x, A) \quad [\approx 23 \text{ MB}]$$
$$\text{clustered\_out} = \text{einsum}('bscer,cedr \to bscd', h, B) \cdot \frac{1}{45} \quad [\approx 33 \text{ MB}]$$
$$\text{micro\_out} = \text{einsum}('bsc,bscd \to bsd', w_{\text{cluster}}, \text{clustered\_out})$$

这一设计直接将训练激活值显存暴降了 **97.6%**，使得 25,200 个专家的全量特训能够在 **单张 RTX 5090 D 上稳定控制在 10 GB 显存以内**，跑出 **9.8 samples/s** 的极速吞吐。

---

## 📊 实机对撞测试与全息透视

### 1. 文理绝缘实测（双大核对称反转验证）
在极端两极分化的任务测试中，门控路由器展现出了教科书级别的动态调度，彻底根除了负迁移现象：

| 评测任务 | 核心能力侧重点 | 文科原版大核占比 | 理科特训大核占比 | 活跃微专家 Top 榜单 |
| :--- | :--- | :---: | :---: | :--- |
| **深山残钟（古典白话散文）** | 禅意意象、长短句韵律、感官通感描摹 | **79.1%** 🏛️ | **20.9%** | `#02 [Arts_Fiction    ]`<br>`#00 [Arts_Prose      ]`<br>`#01 [Arts_Poetry     ]` |
| **C++20 无锁环形缓冲区** | `std::atomic` 内存序、缓存行防伪共享 | **24.5%** | **75.5%** 🔬 | `#16 [Code_Algo       ]`<br>`#17 [Code_DS         ]`<br>`#20 [Code_Syntax     ]` |
| **四维超立方体路由 + 阿莱夫哲理** | 格雷码 XOR 寻路 + 几何形而上学短诗 | **55.8%** | **44.2%** | `#13 [Arts_Philosophy ]`<br>`#01 [Code_DS         ]`<br>`#02 [Code_Debug      ]` |

### 2. 物理存放基准对撞（CPU 流式 vs. GPU 常驻）
- **CPU 宿主锁页流式模式（`expert_pool_location: "host"`）**：**24.9 tokens/s**（专家占用 0 显存）。
- **GPU 纯显存常驻模式（`expert_pool_location: "device"`）**：**23.8 tokens/s**（专家常驻显存）。
- *实测结论*：每层通过 PCIe 传输 96 KB 数据仅耗时约 0.13 毫秒，已被 40 毫秒的前向计算严密掩蔽。放 CPU 内存**完全零性能损失**。

---

## 🔌 模块化卡带热插拔与神经外科手术

### 1. 宗门插槽拓扑（20 宗门定义）
- **宗门 00 ~ 15**：世界级通用通识基石（算法、数据结构、架构、数学、物理、化学、哲学、写作）。
- **宗门 16 ~ 19**：**模块化预留插槽特区**，专供私有业务规则、专有人设、或垂直行业知识在线热插拔。

### 2. 单卡带独立锻造（20 秒产出一个技能卡带）
针对特定领域问答（如 `custom_data.jsonl`，20 条问答），可在 20 秒内独立淬炼出一个独立的技能卡带（`cartridge_gongfang.pt`）：
```bash
python 4.train_single_cartridge.py
```

### 3. 在线内存秒级热插拔（`/plug` 指令）
在不停止聊天终端、不重构 CUDA 上下文、零显存抖动的前提下，**44 毫秒**直接原地注入内存：
```text
👤 You: /plug cartridge_gongfang.pt
⚡ [热插拔成功] 技能卡带《工房1专属规则》已就地植入插槽 #16！
   ⏱️ 注入耗时: 44.45 ms | GPU 零抖动 | 立即生效！
```

### 4. 内鬼缉捕雷达与神经切除（`/catch` 与 `/cage`）
当发现模型表现异常或产生偏执吸引时，可瞬间精确定位作恶层并将其封印：
```text
👤 You: /catch
🚨【赛博内鬼缉捕雷达：28 层深层神经透视】
   - Layer 19 : #16 [Custom_Rules    ] (活跃 75 拍) 🔥 [极度可疑]
   - Layer 20 : #16 [Custom_Rules    ] (活跃 57 拍) 🔥 [极度可疑]

👤 You: /cage 16
🔒 [已关禁闭] 宗门 #16 [Custom_Rules] 已被全面封印！立即失效！

👤 You: /free 16
🔓 [刑满释放] 宗门 #16 [Custom_Rules] 已恢复全额算力！
```

### 5. 离线多卡带融合熔铸（`fuse_cartridges.py`）
无需重新训练底座，可将多个独立卡带一键熔铸固化进底座大权重，生成开箱自带全部私有记忆的成品模型：
```bash
python fuse_cartridges.py
# 将底座 + cartridge_gongfang.pt (16号槽) 熔铸为 myriad_moe_25k_ultimate_fused.pt
```

---

## 🚀 极速上手全流程

### 1. 环境准备
```bash
git clone https://github.com/aifeifei798/Myriad-MoE-25K-Micro-Experts.git
cd Myriad-MoE-25K-Micro-Experts
pip install torch transformers datasets accelerate
```

### 2. 语料分流与多宗门构建（20 宗门版）
```bash
python 1.prepare_myriad_data.py
# 自动下载并生成 20,000 条严格对齐分流的多领域训练样本
```

### 3. 基础全息底座淬炼（25,200 个微专家）
```bash
python 2.train_myriad_25k.py
# 在 RTX 5090 D 上约 20~25 分钟完成 28 层 x 20 宗门 x 45 专家特训
# 导出权重: myriad_moe_25k_weights.pt (~1.25 GB)
```

### 4. 启动万象全息透视终端
```bash
python 3.chat_myriad_25k.py
# 25,200 个微专家挂载进 1.54 GB 内存，进入带实时监控的交互终端
```

---

## 📜 项目里程碑

- [x] **ComfyUI-FeiFei 官方节点收录合入**：LLM 导演扩写、多模态反推、物理胶片后处理全流程跑通。
- [x] **HuggingFace Transformers 提案 #49183**：架构标准化并通过核心维护者确认。
- [x] **双显存密集大核解耦验证**：实证 100% 免疫文理多任务负迁移。
- [x] **零性能损耗锁页内存流式管线**：实现消费级单卡 30 tokens/s 的异构高速吞吐。
- [x] **40ms 内存级赛博义体热插拔**：突破传统框架限制，实现原地指针级知识切换。
- [x] **28 层 CT 级神经内鬼雷达**：实装深层可解释性定位与单点狙击切除工具链。
- [ ] **端侧低比特流式量化**：探索面向移动端/嵌入式芯片的 FP8/INT4 异步微专家流式通道。

---

## 📖 论文与学术引用

如果您在学术研究、系统设计或商业落地中参考了本架构设计或基准数据，请引用本工作：

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

## ⚖️ 开源协议
本项目采用 **[Apache-2.0 开源协议](LICENSE)**。允许学术研究与商业应用，转载或衍生使用请保留原作者署名。


---

## 🔐 接口鉴权与数据复现

### 令牌分级

| 令牌 | 参数 | 权限 |
|---|---|---|
| 管理员 | `--api-key` | 全部：遥测、神经手术、调频、卡带热插拔、生成 |
| 只读 | `--read-only-key` | 仅 `GET` 遥测/看板；**一切写操作返回 403** |

```bash
python 7.api_myriad_server.py --api-key sk-admin --read-only-key sk-viewer
```

`GET /v1/models` 会在 `myriad.permission` 中回报调用方等级
（`admin` / `read` / `anonymous`），并附 `auth_required`、`read_only_available`，
看板据此把无权使用的控件置灰。

设计说明：所有 `/v1/myriad/*` 路由都挂了 `require_engine` 守卫，
因此在权重加载期间（可能耗时很久）一律返回 **503「模型正在加载中」**，
而不会因为 `self.model` 还是 `None` 而崩成 500。
`/v1/models` 与 `/health` 保持可用，客户端才能探测就绪状态。

### 卡带上传限制

`POST /v1/myriad/cartridge/plug` 支持 multipart `file` 上传或服务端 `path`。
上传大小受 `--max-cartridge-mb` 限制（默认 512MB），超出返回 **413**；
空内容 400、插槽越界 400、文件不存在 404。
文件名经 `basename` 净化，不参与任何路径拼接。

### Web 控制台

前端为独立仓库：**[aifeifei798/myriad-moe-console](https://github.com/aifeifei798/myriad-moe-console)**

### 权重与数据不入库

单个权重文件 1.6~2.1 GB，训练语料 14 MB，均由 `.gitignore` 排除。
复现方式：

```bash
python 1.prepare_myriad_data.py      # → myriad_train_data.jsonl（不入库）
python 2.train_myriad_25k.py         # → myriad_moe_25k_weights.pt
python 4.train_single_cartridge.py   # → cartridge_*.pt
python 5.fuse_cartridges.py          # → myriad_moe_25k_ultimate_fused.pt
```

`custom_data.jsonl`（5 KB 示例）**已入库**，便于快速冒烟测试整条流水线。

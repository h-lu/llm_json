# 法律文书提取集成方案 (v2.0 生产级)

本方案专为**百万级法律文书处理**设计，结合了 **DeepSeek (在线)** 的高智能与 **vLLM (本地)** 的高吞吐能力，提供了一套稳健、可扩展的提取系统。

## 1. 核心特性

*   **双模引擎 (Dual-Mode)**:
    *   **ONLINE_API**: 使用 DeepSeek-V3 进行复杂推理或小批量调试。
    *   **LOCAL_OFFLINE**: 使用本地 vLLM 进行海量数据的高吞吐批量推理。
*   **多卡并行 (Multi-GPU)**:
    *   支持**多进程并行**模式。在 7 卡 L40 机器上，可启动 7 个独立进程，吞吐量提升 7 倍。
*   **生产级存储 (Sharded JSONL)**:
    *   采用分片 JSONL (`input_part_001.jsonl`) 存储，大幅提升 I/O 效率。
*   **断点续传 (Checkpointing)**:
    *   内置 ID 级状态追踪。程序中断后重启，会自动跳过已处理的记录，实现秒级恢复。
*   **结构化保证**:
    *   利用 vLLM 的 `guided_json` 技术，在本地推理时也能保证输出严格符合 Pydantic Schema。

## 2. 快速开始

### **2.1 环境准备**
```bash
pip install -r requirements.txt
cp .env.example .env

# 初次使用 Logfire 需要认证（用于日志可观测性）
logfire auth
```

### **2.2 配置 (.env)**
所有配置均在 `.env` 文件中管理。核心参数如下：

```properties
# 运行模式: ONLINE_API (调试) 或 LOCAL_OFFLINE (生产)
EXECUTION_MODE=LOCAL_OFFLINE

# 系统提示词: 可自定义为任何领域的提取任务
# 示例（法律）: 你是一名法律专家助手。请从提供的中国法院判决书中提取结构化数据。
# 示例（医疗）: 你是一名医疗专家助手。请从提供的病历中提取结构化数据。
SYSTEM_PROMPT=你是一名法律专家助手。请从提供的中国法院判决书中提取结构化数据。

# 生成参数
TEMPERATURE=0.0          # 0.0 = 确定性输出
MAX_OUTPUT_TOKENS=2048   # 最大输出长度

# 重试配置（在线 API 模式）
MAX_RETRIES=3            # 最大重试次数
RETRY_DELAY_SECONDS=2    # 重试延迟（秒）

# 数据目录
BATCH_INPUT_DIR=data
BATCH_OUTPUT_DIR=data

# --- 硬件配置 (关键) ---
# 并行 GPU 数量: 设置为显卡总数 (例如 7)
NUM_GPUS=7

# 单卡模型并行度: 建议设为 1 (TP=1)
# 解释: 对于 L40 (48GB) 跑 Qwen2.5-14B，单卡显存足够。
# 启动 7 个 TP=1 的进程比 1 个 TP=4 的进程吞吐量更高。
TENSOR_PARALLEL_SIZE=1
```

### **2.3 数据准备**
系统接受 **JSONL 分片文件** 作为输入。

**方式 A: 迁移现有 .txt 文件**
如果您手头有大量 .txt 文件，请将其放入 `data/` 目录，然后运行迁移脚本：
```bash
python migrate_txt_to_jsonl.py
```
它会将所有 .txt 打包成 `data/input_part_XXX.jsonl`。

**方式 B: 直接准备 JSONL**
格式要求 (每行一个 JSON):
```json
{"custom_id": "unique_case_id_001", "text": "判决书全文内容..."}
```
文件名建议: `input_part_001.jsonl`, `input_part_002.jsonl`...

### **2.4 运行提取**
```bash
python unified_runner.py
```
*   **Master 进程**会自动扫描所有分片，并均匀分配给 7 个 **Worker 进程**。
*   每个 Worker 绑定一张显卡，并行处理。
*   结果实时写入对应的 `output_part_XXX.jsonl`。
*   **随时可以中断 (Ctrl+C)**，下次运行会自动续传。

## 3. 目录结构

```text
.
├── .env                  # 统一配置文件
├── unified_runner.py     # 主程序 (Master-Worker 架构)
├── schemas.py            # 数据结构定义 (Pydantic)
├── migrate_txt_to_jsonl.py # 数据迁移工具
├── data/                 # 数据存储目录
│   ├── input_part_001.jsonl   # 输入分片
│   ├── output_part_001.jsonl  # 提取结果 (自动生成)
│   └── ...
└── requirements.txt
```

## 4. 最佳实践建议

### **4.1 多卡策略**
*   **推荐配置**: `NUM_GPUS=7`, `TENSOR_PARALLEL_SIZE=1`
    *   启动 7 个独立 vLLM 实例，最大化吞吐量。
    *   每个实例独占一张显卡，互不干扰。
*   **不推荐**: `TENSOR_PARALLEL_SIZE=7` 
    *   vLLM 要求 TP 能整除注意力头数，且最好是 2 的幂 (1, 2, 4, 8)。
    *   Qwen 14B 的注意力头数是 40，无法被 7 整除。

### **4.2 性能参数详解**

vLLM 有三个关键参数控制性能和显存占用，理解它们之间的关系非常重要：

#### **参数定义**
1.  **`MAX_MODEL_LEN`** (默认 8192) - 单个序列的最大长度
    *   含义：每条请求（输入+输出）能达到的最大 Token 数。
    *   类比：一本书最多能有多少页。

2.  **`MAX_NUM_SEQS`** (默认 256) - 同时处理的最大序列数
    *   含义：GPU 能**同时**服务多少条请求。
    *   类比：图书馆能同时容纳多少本书。

3.  **`MAX_NUM_BATCHED_TOKENS`** (默认 2048) - 每次前向传播的最大 Token 数
    *   含义：GPU **每次计算**时处理的 Token 总数（所有序列加起来）。
    *   类比：图书管理员一次能翻阅的总页数。

#### **为什么 `MAX_NUM_BATCHED_TOKENS` < `MAX_MODEL_LEN`？**

关键点：**`MAX_NUM_BATCHED_TOKENS` 是所有序列的总和，而 `MAX_MODEL_LEN` 是单个序列的上限。**

**示例 1: 处理 4 条短请求**
```
请求 A: 150 tokens  |  请求 B: 300 tokens
请求 C: 100 tokens  |  请求 D: 200 tokens
----------------------------------------
本批次总计: 750 tokens ✅ (< 2048，可以一起处理)
```

**示例 2: 处理 1 条长请求**
```
请求 E: 8000 tokens (合法，< MAX_MODEL_LEN=8192)

vLLM 使用 "Chunked Prefill" 分块处理:
- 第 1 次前向传播: 处理前 2048 tokens
- 第 2 次前向传播: 处理接下来的 2048 tokens
- ...
- 分 4 次完成，每次不超过 MAX_NUM_BATCHED_TOKENS
```

#### **推荐配置**

**保守配置（稳定优先）**
```properties
MAX_MODEL_LEN=8192           # 单条请求最长 8K
MAX_NUM_BATCHED_TOKENS=2048  # 每次前向传播 2K
MAX_NUM_SEQS=256             # 最多 256 个并发请求
```

**激进配置（吞吐优先，48GB 显存充足时）**
```properties
MAX_MODEL_LEN=8192
MAX_NUM_BATCHED_TOKENS=8192  # 提高到和 MAX_MODEL_LEN 一样
MAX_NUM_SEQS=512             # 增加并发数
```

**如何调优？**
*   如果 GPU 显存占用只有 50-60%，说明还有余量 → 增大 `MAX_NUM_BATCHED_TOKENS`
*   如果经常 OOM (显存不足) → 减小 `MAX_NUM_BATCHED_TOKENS`
*   运行时可通过 `nvidia-smi` 监控显存使用率

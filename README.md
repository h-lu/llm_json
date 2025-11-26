# 法律文书提取系统 (生产级 v2.0)

通用的**大规模文本结构化提取解决方案**，专为百万级数据处理设计。采用双引擎架构 + 生产级可观测性，可广泛应用于法律、医疗、金融等领域的文档智能化。

## ✨ 核心特性

### 🚀 双模引擎
- **在线 API 模式**: 使用 DeepSeek/OpenAI 兼容 API，适合复杂推理和小批量调试
- **本地 vLLM 模式**: 本地 GPU 高吞吐批量推理，适合百万级数据处理

### 📊 生产级可观测性 (Logfire)
- **自动追踪**: 所有 LLM 调用、Pydantic 验证、API 错误自动上传 Logfire
- **实时监控**: Web Dashboard 查看成功率、延迟、Token 消耗
- **零侵入**: 仅需 `logfire auth` 一次认证，无需修改业务代码

### ⚡ 高性能架构
- **多卡并行**: 7 卡 L40 可启动 7 个独立进程，吞吐量提升 7 倍
- **分片存储**: JSONL 分片 + 断点续传，程序中断后秒级恢复
- **自动重试**: API 调用失败自动重试（最多 3 次），提升稳定性

### 🎯 通用化设计
- **配置驱动**: 修改 `.env` 中的 `SYSTEM_PROMPT` 即可切换领域（法律→医疗→财务）
- **Schema 可扩展**: 替换 `schemas.py` 适配任何结构化提取需求
- **零代码切换**: 在线 API ↔ 本地 vLLM 一键切换

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 复制配置文件
cp .env.example .env

# 初次使用 Logfire（生产级监控）
logfire auth
```

**Logfire 认证说明：**
- 首次运行会打开浏览器，使用 GitHub 登录即可
- 认证后所有日志会自动上传到您的 Logfire Dashboard
- 可选：如果不需要云端监控，可跳过此步骤（本地仍会打印日志）

---

### 2. 配置说明

编辑 `.env` 文件，核心参数如下：

```properties
# ==========================================
# 运行模式
# ==========================================
EXECUTION_MODE=ONLINE_API  # 或 LOCAL_OFFLINE

# ==========================================
# 系统提示词（通用化设计 - 可自定义领域）
# ==========================================
# 法律领域示例
SYSTEM_PROMPT=你是一名法律专家助手。请从提供的中国法院判决书中提取结构化数据。

# 医疗领域示例（取消注释使用）
# SYSTEM_PROMPT=你是一名医疗专家助手。请从提供的病历中提取结构化数据。

# 财务领域示例（取消注释使用）
# SYSTEM_PROMPT=你是一名财务专家助手。请从提供的财务报表中提取结构化数据。

# ==========================================
# 生成参数
# ==========================================
TEMPERATURE=0.0          # 0.0 = 确定性输出（数据提取推荐）
MAX_OUTPUT_TOKENS=2048   # 最大输出长度

# ==========================================
# 重试配置（在线 API 模式）
# ==========================================
MAX_RETRIES=3            # 最大重试次数（防止临时网络错误）
RETRY_DELAY_SECONDS=2    # 重试延迟（秒）

# ==========================================
# 数据目录
# ==========================================
BATCH_INPUT_DIR=data
BATCH_OUTPUT_DIR=data

# ==========================================
# 硬件配置（本地 vLLM 模式）
# ==========================================
NUM_GPUS=7               # 并行 GPU 数量
TENSOR_PARALLEL_SIZE=1   # 单卡模型并行度（推荐 TP=1）

# L40 (48GB) 运行 Qwen 14B 建议配置：
# - NUM_GPUS=7, TP=1  → 启动 7 个独立进程（吞吐量最高）
# - NUM_GPUS=1, TP=4  → 单进程 4 卡并行（模型太大时使用）
```

---

### 3. 数据准备

系统接受 **JSONL 分片文件** 作为输入，格式如下：

```jsonl
{"custom_id": "doc_001", "text": "这里是原始文本内容..."}
{"custom_id": "doc_002", "text": "第二条文档..."}
```

**从 TXT 文件迁移：**
```bash
# 如果您的数据是 .txt 文件，使用迁移脚本转换
python migrate_txt_to_jsonl.py
```

**手动创建分片：**
```bash
# 创建输入文件
echo '{"custom_id": "test_001", "text": "测试文本"}' > data/input_part_001.jsonl
```

---

### 4. 运行系统

```bash
python unified_runner.py
```

**运行后输出示例（带 Logfire）：**
```
Logfire project URL: https://logfire-us.pydantic.dev/your-project
>>> 正在以 [ONLINE_API] 模式运行 (Model: deepseek-chat)...
扫描到 10 个输入分片。
20:36:00.123 处理分片: input_part_001.jsonl
20:36:16.257 Pydantic JudgmentExtraction validate_json  ← Logfire 自动追踪
20:36:31.189 Chat Completion with 'deepseek-chat' [LLM] ← Logfire 自动追踪
20:36:41.094 分片完成: input_part_001.jsonl - 成功: 1000/1000
```

**查看实时监控：**
访问 Logfire Dashboard (控制台会输出链接)，可查看：
- 📈 API 调用成功率和延迟
- 💰 Token 消耗统计
- 🔍 每个请求的完整追踪链路
- ⚠️ 错误详情和堆栈

---

## 📖 进阶配置

### vLLM 性能参数

```properties
# 显存占用率 (0.90 - 0.95)
GPU_MEMORY_UTILIZATION=0.90

# 上下文长度限制（防止 OOM）
MAX_MODEL_LEN=8192

# 开启前缀缓存（加速 System Prompt）
ENABLE_PREFIX_CACHING=True

# 高级调优（可选）
MAX_NUM_SEQS=256              # 最大并发序列数
MAX_NUM_BATCHED_TOKENS=2048   # 每次迭代处理的最大 Token 数
```

**参数说明：**
- `MAX_MODEL_LEN`: 模型支持的最大上下文长度，建议设为 8192（Qwen 14B）
- `MAX_NUM_SEQS`: 并发处理的序列数，显存越大可设越高
- `MAX_NUM_BATCHED_TOKENS`: 批处理大小，遇到 OOM 可调小

---

## 🔍 Logfire 可观测性详解

### 自动追踪的内容

Logfire 会自动记录以下信息（无需额外代码）：

1. **LLM API 调用**
   - 请求时间、延迟、Token 消耗
   - 输入 Prompt 和输出结果
   - 重试次数和失败原因

2. **Pydantic 验证**
   - Schema 验证成功/失败
   - 验证错误详情
   - 数据类型转换过程

3. **系统指标**
   - 成功率、错误率
   - 平均延迟、P95/P99 延迟
   - 吞吐量统计

### Dashboard 功能

访问您的 Logfire 项目（链接在启动时显示），可以：

- 📊 **实时监控**: 查看当前正在运行的任务
- 🔍 **Trace 查询**: 搜索特定 `custom_id` 的处理记录
- 📈 **性能分析**: 查看延迟分布、Token 消耗趋势
- ⚠️ **错误追踪**: 快速定位失败原因

**示例查询（Dashboard 中使用）：**
```sql
-- 查询失败率最高的 10 个错误
SELECT error_type, count(*) as cnt
FROM records
WHERE status = 'failed'
GROUP BY error_type
ORDER BY cnt DESC
LIMIT 10
```

---

## 📊 性能参考

### 实测数据 (单卡 L40 + Qwen 14B)

| 指标 | 数值 |
|------|------|
| 吞吐量 | ~100 条/分钟 |
| 平均延迟 | ~15 秒/条 |
| 显存占用 | ~35GB / 48GB |
| Token/秒 | ~150 tokens/s |

### 多卡扩展性

| GPU 数量 | 吞吐量 | 处理 100 万条耗时 |
|----------|--------|-------------------|
| 1 卡 | 100 条/分 | ~7 天 |
| 4 卡 | 400 条/分 | ~1.7 天 |
| 7 卡 | 700 条/分 | ~1 天 |

---

## 🛠️ 自定义 Schema

修改 `schemas.py` 以适配您的数据结构：

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class YourCustomSchema(BaseModel):
    """自定义提取结构"""
    field1: str = Field(..., description="字段1的描述（帮助 LLM 理解）")
    field2: Optional[int] = Field(None, description="可选字段")
    nested_list: List[str] = Field(default_factory=list, description="列表字段")
```

**然后在 `unified_runner.py` 中替换：**
```python
from schemas import YourCustomSchema  # 替换 JudgmentExtraction

# ...
response_model=YourCustomSchema  # 替换原有的 response_model
```

---

## 🔧 故障排查

### 1. Logfire 认证失败
```bash
# 手动认证
logfire auth

# 或跳过 Logfire（仅使用本地日志）
# 注释掉 unified_runner.py 中的 logfire.configure()
```

### 2. vLLM OOM 错误
```properties
# 降低显存占用率
GPU_MEMORY_UTILIZATION=0.85

# 减小上下文长度
MAX_MODEL_LEN=4096

# 减小批处理大小
MAX_NUM_BATCHED_TOKENS=1024
```

### 3. API 限流错误
```properties
# 增加重试次数
MAX_RETRIES=5

# 延长重试延迟
RETRY_DELAY_SECONDS=5
```

---

## 📚 文件结构

```
llm_json/
├── unified_runner.py          # 主程序（双模引擎）
├── schemas.py                 # 数据结构定义（Pydantic）
├── migrate_txt_to_jsonl.py    # TXT 文件迁移工具
├── requirements.txt           # 依赖列表
├── .env.example               # 配置模板
├── .env                       # 实际配置（不提交到 Git）
├── .gitignore                 # Git 忽略规则
└── data/                      # 数据目录
    ├── input_part_*.jsonl     # 输入分片
    └── output_part_*.jsonl    # 输出分片
```

---

## 🎓 最佳实践

1. **开发阶段**: 使用 `ONLINE_API` 模式 + 小数据集验证 Schema
2. **生产部署**: 切换到 `LOCAL_OFFLINE` 模式 + vLLM 多卡并行
3. **监控运维**: 定期查看 Logfire Dashboard，关注错误率和延迟
4. **数据安全**: 确保 `.env` 已加入 `.gitignore`，避免泄露 API Key

---

## 📝 License

MIT License

---

## 🙏 致谢

本项目基于以下优秀开源项目：
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [Instructor](https://github.com/instructor-ai/instructor) - 结构化输出抽取
- [Logfire](https://logfire.pydantic.dev/) - Python 原生可观测性平台
- [Pydantic](https://github.com/pydantic/pydantic) - 数据验证框架

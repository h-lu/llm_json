import os
import json
import time
import glob
import multiprocessing
from typing import List, Dict, Set, Any
from dotenv import load_dotenv
from schemas import JudgmentExtraction
import instructor
from openai import OpenAI
import logfire
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from openai import RateLimitError, APIError

# 加载 .env
load_dotenv()

# 配置 Logfire (仅在 master 进程中初始化一次)
logfire.configure()
logfire.instrument_pydantic()

# 读取配置
EXECUTION_MODE = os.getenv("EXECUTION_MODE", "ONLINE_API").upper()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "你是一个有用的助手。")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
BATCH_INPUT_DIR = os.getenv("BATCH_INPUT_DIR", "data")
BATCH_OUTPUT_DIR = os.getenv("BATCH_OUTPUT_DIR", "data")
ONLINE_MODEL_NAME = os.getenv("ONLINE_MODEL_NAME", "deepseek-chat")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RETRY_DELAY_SECONDS", "2"))

def get_input_shards() -> List[str]:
    """获取所有输入分片文件"""
    pattern = os.path.join(BATCH_INPUT_DIR, "input_part_*.jsonl")
    files = sorted(glob.glob(pattern))
    print(f"扫描到 {len(files)} 个输入分片。")
    return files

def get_output_path(input_path: str) -> str:
    """生成输出文件路径"""
    basename = os.path.basename(input_path)
    output_name = basename.replace("input_", "output_")
    return os.path.join(BATCH_OUTPUT_DIR, output_name)

def load_checkpoint(output_path: str) -> Set[str]:
    """加载断点"""
    processed_ids = set()
    if not os.path.exists(output_path):
        return processed_ids
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if "custom_id" in record:
                        processed_ids.add(record["custom_id"])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取断点失败: {e}")
    return processed_ids

def append_results(output_path: str, results: List[Dict]):
    """追加写入结果"""
    with open(output_path, "a", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

# ==========================================
# Worker Process Logic (独立进程)
# ==========================================
def worker_process(gpu_id: int, shard_paths: List[str]):
    """Worker 进程：绑定 GPU 并处理分配的分片"""
    # 1. 设置可见 GPU (必须在导入 vLLM 之前)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[Worker {gpu_id}] 启动，绑定 GPU {gpu_id}，待处理分片数: {len(shard_paths)}")
    
    if not shard_paths:
        print(f"[Worker {gpu_id}] 无任务，退出。")
        return

    # 2. 初始化 vLLM (每个进程独立初始化)
    try:
        from vllm import LLM, SamplingParams
        
        # 读取配置 (注意: 这里是进程内的环境变量，或者重新读取 .env)
        model_path = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-14B-Instruct")
        # Worker 内部通常 TP=1
        tp_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1")) 
        gpu_util = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
        max_len = int(os.getenv("MAX_MODEL_LEN", "8192"))
        prefix_caching = os.getenv("ENABLE_PREFIX_CACHING", "True").lower() == "true"

        print(f"[Worker {gpu_id}] 初始化 vLLM: {model_path} (TP={tp_size})...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            enable_prefix_caching=prefix_caching,
            trust_remote_code=True,
        )
        
        json_schema = JudgmentExtraction.model_json_schema()
        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            guided_json=json_schema 
        )
        
    except Exception as e:
        print(f"[Worker {gpu_id}] 初始化失败: {e}")
        return

    # 3. 顺序处理分配的分片
    for input_path in shard_paths:
        output_path = get_output_path(input_path)
        print(f"[Worker {gpu_id}] 开始处理: {os.path.basename(input_path)}")
        
        processed_ids = load_checkpoint(output_path)
        
        # 读取待处理数据
        pending_records = []
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record["custom_id"] not in processed_ids:
                            pending_records.append(record)
                    except: continue
        except Exception as e:
            print(f"[Worker {gpu_id}] 读取文件失败 {input_path}: {e}")
            continue
            
        if not pending_records:
            print(f"[Worker {gpu_id}] {os.path.basename(input_path)} 已完成，跳过。")
            continue

        # 分批推理
        total = len(pending_records)
        for i in range(0, total, BATCH_SIZE):
            batch = pending_records[i : i + BATCH_SIZE]
            prompts = [
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{rec['text']}<|im_end|>\n<|im_start|>assistant\n"
                for rec in batch
            ]
            
            outputs = llm.generate(prompts, sampling_params)
            
            results_to_save = []
            for j, output in enumerate(outputs):
                rec = batch[j]
                try:
                    json_str = output.outputs[0].text
                    obj = JudgmentExtraction.model_validate_json(json_str)
                    results_to_save.append({
                        "custom_id": rec["custom_id"],
                        "extraction": obj.model_dump(),
                        "status": "success"
                    })
                except Exception as e:
                    results_to_save.append({
                        "custom_id": rec["custom_id"],
                        "error": str(e),
                        "raw_output": output.outputs[0].text,
                        "status": "failed"
                    })
            
            append_results(output_path, results_to_save)
            print(f"[Worker {gpu_id}] 进度: {min(i + BATCH_SIZE, total)}/{total}")

    print(f"[Worker {gpu_id}] 所有任务完成。")


# ==========================================
# Master Process Logic
# ==========================================
def run_local_offline_multi_gpu():
    """模式 B: 本地离线 (多 GPU 并行)"""
    print(f">>> 正在以 [LOCAL_OFFLINE] 模式运行 (GPUs={NUM_GPUS})...")
    
    input_shards = get_input_shards()
    if not input_shards:
        print("未找到输入分片，请先运行 migrate_txt_to_jsonl.py")
        return

    # 1. 分发任务 (Round-Robin)
    worker_tasks = [[] for _ in range(NUM_GPUS)]
    for i, shard in enumerate(input_shards):
        worker_id = i % NUM_GPUS
        worker_tasks[worker_id].append(shard)

    # 2. 启动 Worker 进程
    processes = []
    # 必须使用 'spawn' 启动方式以兼容 CUDA
    ctx = multiprocessing.get_context("spawn")
    
    for gpu_id in range(NUM_GPUS):
        tasks = worker_tasks[gpu_id]
        if not tasks:
            print(f"警告: GPU {gpu_id} 没有分配到任务。")
            continue
            
        p = ctx.Process(target=worker_process, args=(gpu_id, tasks))
        p.start()
        processes.append(p)
    
    # 3. 等待所有进程结束
    for p in processes:
        p.join()
    
    print("所有 Worker 已完成。")

def run_online_api():
    """模式 A: 在线 API (单进程)"""
    # ... (保持原有逻辑不变，为了简洁省略部分重复代码，实际使用时应保留)
    print(f">>> 正在以 [ONLINE_API] 模式运行 (Model: {ONLINE_MODEL_NAME})...")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    
    if not api_key or "placeholder" in api_key:
        logfire.error("错误: 请在 .env 中配置 DEEPSEEK_API_KEY")
        return

    # 创建 OpenAI 客户端并添加 Logfire 监控
    openai_client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(openai_client)
    client = instructor.from_openai(openai_client)  # 不在这里设置 max_retries

    input_shards = get_input_shards()
    for shard in input_shards:
        output_path = get_output_path(shard)
        logfire.info(f"处理分片: {os.path.basename(shard)}")
        
        processed_ids = load_checkpoint(output_path)
        total_count = 0
        success_count = 0
        
        with open(shard, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    custom_id = record.get("custom_id")
                    if custom_id in processed_ids:
                        continue
                    
                    # 使用 instructor 内置的重试机制（更简洁）
                    extraction = client.chat.completions.create(
                        model=ONLINE_MODEL_NAME,
                        response_model=JudgmentExtraction,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": record["text"]}
                        ],
                        temperature=TEMPERATURE,
                        max_retries=MAX_RETRIES,  # 在这里设置重试
                    )
                    
                    append_results(output_path, [{
                        "custom_id": custom_id,
                        "extraction": extraction.model_dump(),
                        "status": "success"
                    }])
                    success_count += 1
                    total_count += 1
                    
                except Exception as e:
                    logfire.error(f"处理失败 [{custom_id}]: {e}")
                    append_results(output_path, [{
                        "custom_id": custom_id,
                        "error": str(e),
                        "status": "failed"
                    }])
                    total_count += 1
        
        logfire.info(f"分片完成: {os.path.basename(shard)} - 成功: {success_count}/{total_count}")

if __name__ == "__main__":
    if EXECUTION_MODE == "ONLINE_API":
        run_online_api()
    elif EXECUTION_MODE == "LOCAL_OFFLINE":
        run_local_offline_multi_gpu()
    else:
        print(f"未知的运行模式: {EXECUTION_MODE}")

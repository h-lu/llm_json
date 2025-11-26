import os
import json
import glob
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("BATCH_INPUT_DIR", "data")
SHARD_SIZE = int(os.getenv("SHARD_SIZE", "10000"))

def migrate():
    print(f"开始迁移 {DATA_DIR} 下的 .txt 文件到 JSONL 分片...")
    
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not txt_files:
        print("未找到 .txt 文件，无需迁移。")
        return

    current_shard_idx = 1
    current_shard_data = []
    
    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 使用文件名作为 custom_id，或者生成 UUID
            file_id = os.path.basename(fpath)
            
            record = {
                "custom_id": file_id,
                "text": content
            }
            current_shard_data.append(record)
            
            # 如果达到分片大小，写入文件
            if len(current_shard_data) >= SHARD_SIZE:
                write_shard(current_shard_idx, current_shard_data)
                current_shard_idx += 1
                current_shard_data = []
                
        except Exception as e:
            print(f"读取文件 {fpath} 失败: {e}")

    # 写入剩余数据
    if current_shard_data:
        write_shard(current_shard_idx, current_shard_data)

    print("迁移完成！您可以删除旧的 .txt 文件了。")

def write_shard(idx, data):
    filename = os.path.join(DATA_DIR, f"input_part_{idx:03d}.jsonl")
    print(f"写入分片: {filename} ({len(data)} 条记录)...")
    with open(filename, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    migrate()

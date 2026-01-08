import os
import json
from tqdm import tqdm
from llama_cpp import Llama

from processing import *
from jurex_qwen import *

# 模型路径
model_path = "E:/Data/models/qwen1.5-7b-chat-q4_k_m.gguf"

# 加载模型，使用GPU加速
print("加载模型中...")
model = Llama(
    model_path=model_path,
    n_ctx=32768,
    n_gpu_layers=-1,  # 所有层都放到GPU
    n_batch=512,  # 增加批处理大小
    n_threads=8,  # CPU线程
    offload_kqv=True,  # 卸载K,Q,V到GPU
    verbose=False
)

print("模型加载完成!")


def process_single(ridx, cid):
    """处理单个候选"""
    # 1. 加载数据
    try:
        c_dict = getCandidateDict(ridx, cid)
    except:
        return False

    # 2. 检查des属性是否已存在且不为空
    if 'des' in c_dict and c_dict['des']:
        # des已存在且不为空，跳过
        return True  # 返回True表示跳过

    # 3. 获取犯罪列表
    crime_list = getCandidateCrimes(ridx, cid)
    if not crime_list:
        c_dict['des'] = {}
        save_result(ridx, cid, c_dict)
        return True

    # 4. 处理每个犯罪
    c_dict['des'] = {}
    for crime in crime_list:
        # 准备prompt
        text = c_dict.get('ajjbqk', '') + c_dict.get('cpfxgc', '')
        prompt = f"请分析以下案例是否构成{crime}：{text}"

        # 生成响应
        response = model(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            stop=["<|endoftext|>", "</s>"]
        )

        # 解析响应
        if response and 'choices' in response:
            analysis = response['choices'][0]['text']
            c_dict['des'][crime] = {"analysis": analysis}

    # 5. 保存结果
    save_result(ridx, cid, c_dict)
    return True


def save_result(ridx, cid, c_dict):
    """保存结果"""
    save_dir = f"E:/Py_Dev/IceBerg/data/candidates/{ridx}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{cid}.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(c_dict, f, ensure_ascii=False, indent=2)


# 主循环
ROOT_PATH = "E:/Py_Dev/IceBerg/data/candidates"
all_ridx = os.listdir(ROOT_PATH)

total_processed = 0
total_skipped = 0

for ridx in all_ridx:
    ridx_path = os.path.join(ROOT_PATH, ridx)
    if not os.path.isdir(ridx_path):
        continue

    # 获取所有json文件
    cid_files = [f for f in os.listdir(ridx_path) if f.endswith('.json')]

    for file in tqdm(cid_files, desc=f"处理 {ridx}"):
        cid = file.replace('.json', '')

        # 检查是否已处理
        file_path = os.path.join(ridx_path, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'des' in data and data['des']:
                    total_skipped += 1
                    continue
        except:
            pass

        # 处理候选
        success = process_single(ridx, cid)
        if success:
            total_processed += 1

        # 每处理10个打印一次进度
        if total_processed % 10 == 0:
            print(f"已处理: {total_processed}, 跳过: {total_skipped}")

print(f"\n处理完成!")
print(f"总计处理: {total_processed}")
print(f"总计跳过: {total_skipped}")
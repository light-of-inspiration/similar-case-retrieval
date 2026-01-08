import os
import json
import sys
import time
import gc
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from processing import *
from jurex_qwen import *


# === 设备配置 ===
def setup_torch_device():
    """设置PyTorch CUDA设备"""
    print("=" * 60)
    print("PyTorch CUDA设备设置")
    print("=" * 60)

    if torch.cuda.is_available():
        # 选择GPU设备
        device = torch.device("cuda:0")  # 明确指定第一个GPU

        # 获取设备信息
        print(f"✅ CUDA可用，使用设备: {device}")
        print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   PyTorch版本: {torch.__version__}")

        # 设置PyTorch优化参数
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32，加速矩阵乘法
        torch.backends.cudnn.allow_tf32 = True  # 允许TF32

        # 设置设备为当前设备
        torch.cuda.set_device(device)

        # 验证设备
        print(f"   当前设备: {torch.cuda.current_device()}")
        print(f"   设备数量: {torch.cuda.device_count()}")

        return device
    else:
        print("❌ CUDA不可用，使用CPU")
        return torch.device("cpu")


# 初始化设备
device = setup_torch_device()


# === llama_cpp集成torch设备 ===
class TorchEnhancedLlama:
    """增强版Llama模型，集成PyTorch设备管理"""

    def __init__(self, model_path, device=device):
        self.device = device
        self.model_path = model_path

        # 导入llama_cpp
        try:
            from llama_cpp import Llama
            self.Llama = Llama
        except ImportError:
            print("❌ 请安装llama-cpp-python: pip install llama-cpp-python")
            sys.exit(1)

        # 加载模型
        self.model = self._load_model()

        # 创建CUDA张量用于监控
        self._init_cuda_monitor()

    def _load_model(self):
        """加载模型并配置GPU参数"""
        print(f"\n🚀 加载模型: {os.path.basename(self.model_path)}")

        # 根据设备类型配置参数
        if self.device.type == "cuda":
            n_gpu_layers = -1  # 所有层都在GPU
            n_batch = 2048  # 增加批处理大小
            n_threads = 8  # CPU线程数
        else:
            n_gpu_layers = 0
            n_batch = 512
            n_threads = 16

        config = {
            'model_path': self.model_path,
            'n_ctx': 32768,
            'n_gpu_layers': n_gpu_layers,
            'n_batch': n_batch,
            'n_threads': n_threads,
            'offload_kqv': True,
            'flash_attn': True,
            'use_mmap': True,
            'use_mlock': False,
            'verbose': False
        }

        print(f"   配置: GPU层数={n_gpu_layers}, 批大小={n_batch}, 线程数={n_threads}")

        try:
            model = self.Llama(**config)
            print("✅ 模型加载成功!")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 尝试简化配置
            print("尝试简化配置...")
            try:
                model = self.Llama(
                    model_path=self.model_path,
                    n_ctx=8192,
                    n_gpu_layers=n_gpu_layers,
                    verbose=False
                )
                print("✅ 模型加载成功（简化配置）!")
                return model
            except Exception as e2:
                print(f"❌ 简化配置也失败: {e2}")
                raise

    def _init_cuda_monitor(self):
        """初始化CUDA监控张量"""
        if self.device.type == "cuda":
            # 创建一些CUDA张量用于监控GPU使用
            self.monitor_tensor = torch.zeros(1000, 1000, device=self.device)
            print(f"✅ CUDA监控张量创建在: {self.monitor_tensor.device}")

    def create_completion(self, prompt, **kwargs):
        """创建完成，集成GPU监控"""
        # 在GPU上执行推理
        if self.device.type == "cuda":
            # 确保监控GPU使用
            self._log_gpu_memory("推理前")

        # 调用原始模型
        result = self.model.create_completion(prompt, **kwargs)

        if self.device.type == "cuda":
            self._log_gpu_memory("推理后")

        return result

    def _log_gpu_memory(self, stage):
        """记录GPU内存使用"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 3

            print(f"   [{stage}] GPU内存: 分配={allocated:.2f}GB, 保留={reserved:.2f}GB")


# === GPU加速数据处理 ===
class CUDADataProcessor:
    """CUDA加速的数据处理器"""

    def __init__(self, device):
        self.device = device
        self.data_cache = {}  # 缓存数据到GPU

    def process_text_batch(self, texts, max_length=512):
        """批量处理文本到CUDA张量"""
        if not texts:
            return None

        # 创建批次
        batch_size = len(texts)

        # 将文本编码为数字（简化示例，实际需要tokenizer）
        encoded_texts = [self._simple_encode(text, max_length) for text in texts]

        # 转换为PyTorch张量并移动到CUDA
        tensor_batch = torch.tensor(encoded_texts, dtype=torch.long)

        # 使用torch.to(device)移动到GPU
        tensor_batch = tensor_batch.to(self.device)

        print(f"✅ 数据批次已移动到 {self.device}: {tensor_batch.shape}")

        return tensor_batch

    def _simple_encode(self, text, max_length):
        """简单的文本编码（实际应用应使用合适的tokenizer）"""
        # 简化的编码：将字符转换为ASCII码
        encoded = [ord(c) for c in text[:max_length]]

        # 填充或截断
        if len(encoded) < max_length:
            encoded += [0] * (max_length - len(encoded))
        else:
            encoded = encoded[:max_length]

        return encoded

    def cache_to_gpu(self, key, data):
        """缓存数据到GPU"""
        if isinstance(data, (list, np.ndarray)):
            # 转换为PyTorch张量
            tensor = torch.tensor(data)
            # 使用torch.to(device)移动到GPU
            tensor = tensor.to(self.device)
            self.data_cache[key] = tensor
            print(f"✅ 数据已缓存到GPU: {key} -> {tensor.shape}")
            return tensor
        return data


# === 批量推理处理器 ===
class CUDABatchProcessor:
    """CUDA加速的批量处理器"""

    def __init__(self, llama_model, device, batch_size=4):
        self.llama_model = llama_model
        self.device = device
        self.batch_size = batch_size

        # 创建CUDA数据处理器
        self.data_processor = CUDADataProcessor(device)

        # 统计信息
        self.stats = {
            'processed': 0,
            'batches': 0,
            'gpu_time': 0.0,
            'total_tokens': 0
        }

    def batch_generate(self, prompts, max_tokens=512, temperature=0.7):
        """批量生成，使用CUDA加速"""
        all_responses = []

        # 记录开始时间
        start_time = time.time()

        # 分批处理
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            self.stats['batches'] += 1

            # 在GPU上预处理数据
            with torch.cuda.device(self.device):
                # 可以在这里添加数据预处理步骤
                # 例如：将文本转换为CUDA张量

                # 执行推理
                batch_start = time.time()
                batch_responses = self._process_batch(batch_prompts, max_tokens, temperature)
                batch_time = time.time() - batch_start

                self.stats['gpu_time'] += batch_time
                self.stats['total_tokens'] += sum(len(p) for p in batch_prompts)

                all_responses.extend(batch_responses)

                # 监控GPU使用
                self._monitor_gpu(f"批次 {i // self.batch_size + 1}")

        total_time = time.time() - start_time
        print(f"📊 批次处理完成: {self.stats['batches']}批次, "
              f"GPU时间: {self.stats['gpu_time']:.2f}秒, "
              f"总时间: {total_time:.2f}秒")

        return all_responses

    def _process_batch(self, prompts, max_tokens, temperature):
        """处理单个批次"""
        responses = []

        for prompt in prompts:
            try:
                # 使用llama模型生成
                response = self.llama_model.create_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stop=["<|endoftext|>", "</s>", "###"]
                )

                if response and 'choices' in response:
                    text = response['choices'][0]['text']
                    responses.append(text)
                else:
                    responses.append("")

            except Exception as e:
                print(f"❌ 推理失败: {e}")
                responses.append("")

        return responses

    def _monitor_gpu(self, label=""):
        """监控GPU使用情况"""
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 3

            # 获取GPU利用率（需要pynvml）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index or 0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                print(f"   {label} GPU使用: {allocated:.2f}GB, 利用率: {util.gpu}%")
            except:
                print(f"   {label} GPU内存: {allocated:.2f}GB")

            # 如果内存使用过高，清理缓存
            if allocated > 18:  # 超过18GB
                torch.cuda.empty_cache()


# === 候选数据处理 ===
class CandidateProcessor:
    """候选数据处理器，包含des属性检查"""

    def __init__(self, batch_processor):
        self.batch_processor = batch_processor
        self.device = batch_processor.device

        # 导入数据加载函数
        try:
            from processing import getCandidateDict, getCandidateCrimes
            self.getCandidateDict = getCandidateDict
            self.getCandidateCrimes = getCandidateCrimes
        except ImportError:
            print("❌ 无法导入processing模块")
            raise

    def check_des_exists(self, c_dict):
        """检查des属性是否已存在且不为空"""
        if not c_dict:
            return True

        # 检查des字段是否存在
        if 'des' not in c_dict:
            return False

        des_value = c_dict['des']

        # 检查des是否为空
        if not des_value:
            return False

        # 检查des是否为有效字典且有内容
        if isinstance(des_value, dict) and len(des_value) > 0:
            return True

        # 检查是否为其他非空值
        if des_value:
            return True

        return False

    def generate_prompt(self, c_dict, crime):
        """生成提示词"""
        text = c_dict.get('ajjbqk', '') + c_dict.get('cpfxgc', '')

        prompt = f"""你是一个专业的法律分析专家。请分析以下案例是否构成【{crime}】罪名：

【案件事实】
{text}

请从以下三个维度进行分析：
1. 罪名分析：是否构成该罪名，为什么
2. 构成要件：主体、客体、主观方面、客观方面
3. 量刑情节：从重、从轻、减轻或免除处罚情节

请用JSON格式输出分析结果："""

        return prompt

    def process_candidate(self, ridx, cid):
        """处理单个候选"""
        try:
            # 加载候选数据
            c_dict = self.getCandidateDict(ridx, cid)

            # 检查des属性
            if self.check_des_exists(c_dict):
                return None, "已存在des属性"

            # 获取犯罪列表
            crime_list = self.getCandidateCrimes(ridx, cid)
            if not crime_list:
                c_dict['des'] = {}
                return c_dict, "无犯罪列表"

            # 为每个犯罪生成提示词
            prompts = []
            crimes = []

            for crime in crime_list:
                prompt = self.generate_prompt(c_dict, crime)
                prompts.append(prompt)
                crimes.append(crime)

            # 批量推理（使用CUDA加速）
            print(f"  正在处理 {len(prompts)} 个提示词...")
            responses = self.batch_processor.batch_generate(prompts)

            # 解析响应并构建des字典
            c_dict['des'] = {}
            for crime, response in zip(crimes, responses):
                # 解析响应（这里简化处理，实际需要更复杂的解析）
                crime_desc = self._parse_response(response)
                c_dict['des'][crime] = crime_desc

            return c_dict, "成功"

        except Exception as e:
            print(f"❌ 处理候选 {ridx}/{cid} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None, str(e)

    def _parse_response(self, response):
        """解析响应"""
        # 简化的解析，实际应根据模型输出格式调整
        if not response:
            return {"error": "空响应"}

        # 尝试提取JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass

        # 如果不是JSON，返回原始文本
        return {"analysis": response[:500]}  # 限制长度


# === 主程序 ===
def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("本地GGUF模型 + PyTorch CUDA加速系统")
    print("=" * 70)

    # 1. 设置CUDA设备
    device = setup_torch_device()

    # 2. 查找本地模型
    model_dir = Path("E:/Data/models/")
    gguf_models = list(model_dir.glob("*.gguf"))

    if not gguf_models:
        print("❌ 没有找到GGUF模型文件")
        return

    print(f"\n📂 找到 {len(gguf_models)} 个本地模型:")
    for i, model_path in enumerate(gguf_models):
        print(f"  [{i + 1}] {model_path.name} ({model_path.stat().st_size / 1024 ** 3:.2f} GB)")

    # 选择第一个模型
    selected_model = gguf_models[0]
    print(f"\n✅ 选择模型: {selected_model.name}")

    # 3. 加载模型
    try:
        llama_model = TorchEnhancedLlama(str(selected_model), device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 创建CUDA加速的批处理器
    batch_processor = CUDABatchProcessor(llama_model, device, batch_size=4)

    # 5. 创建候选处理器
    candidate_processor = CandidateProcessor(batch_processor)

    # 6. 处理数据
    ROOT_PATH = Path(r"E:/Py_Dev/IceBerg/data/candidates")

    if not ROOT_PATH.exists():
        print(f"❌ 数据目录不存在: {ROOT_PATH}")
        return

    # 获取所有ridx目录
    all_ridx = [d for d in ROOT_PATH.iterdir() if d.is_dir()]

    print(f"\n📁 找到 {len(all_ridx)} 个候选目录")

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # 处理每个目录
    for ridx_dir in all_ridx:
        ridx = ridx_dir.name
        print(f"\n📂 处理目录: {ridx}")

        # 获取所有JSON文件
        json_files = list(ridx_dir.glob("*.json"))

        if not json_files:
            print(f"   没有JSON文件")
            continue

        print(f"   找到 {len(json_files)} 个候选文件")

        # 处理每个文件
        for json_file in tqdm(json_files, desc=f"处理 {ridx}"):
            cid = json_file.stem  # 去掉扩展名

            # 处理候选
            result, status = candidate_processor.process_candidate(ridx, cid)

            if status == "已存在des属性":
                total_skipped += 1
                continue
            elif status == "成功":
                # 保存结果
                try:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    total_processed += 1
                except Exception as e:
                    print(f"❌ 保存文件失败 {json_file}: {e}")
                    total_errors += 1
            else:
                total_errors += 1

            # 定期清理GPU缓存
            if total_processed % 10 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

    # 7. 打印统计信息
    print("\n" + "=" * 70)
    print("🎉 处理完成!")
    print("=" * 70)
    print(f"   总计处理: {total_processed}")
    print(f"   总计跳过: {total_skipped}")
    print(f"   总计错误: {total_errors}")

    # 8. 最终GPU状态
    if device.type == "cuda":
        print("\n📊 最终GPU状态:")
        print(f"   已分配内存: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        print(f"   保留内存: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        # 清理
        torch.cuda.empty_cache()
        print("✅ GPU缓存已清理")


# === 脚本入口 ===
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在清理...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ 清理完成")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback

        traceback.print_exc()

        # 确保清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
# environment_fix.py
import os
import sys
import subprocess
import warnings


def fix_mkl_issue():
    """修复 MKL 相关问题"""
    print("正在修复 MKL 环境问题...")

    # 设置环境变量
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 忽略相关警告
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*mkl_intel_thread.*")
    warnings.filterwarnings("ignore", message=".*Intel MKL.*")

    print("环境修复完成")


def check_environment():
    """检查环境状态"""
    print("检查环境状态...")

    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        print(f"✓ CUDA 可用: {torch.cuda.is_available()}")

        import numpy as np
        print(f"✓ NumPy 版本: {np.__version__}")

        # 测试基本运算
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        result = np.dot(a, b)
        print(f"✓ NumPy 运算测试: {a} · {b} = {result}")

        return True
    except Exception as e:
        print(f"✗ 环境检查失败: {e}")
        return False


if __name__ == "__main__":
    fix_mkl_issue()
    check_environment()
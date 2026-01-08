import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 设置中文字体（如果分析中文文本）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimplePurpleAnalyzer:
    """简化但美观的紫色配色分析器"""

    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def get_purple_colors(self, n_colors=10):
        """获取紫色渐变颜色"""
        base_colors = sns.color_palette("husl", n_colors)
        # 转换为紫色系
        purple_colors = []
        for color in base_colors:
            # 增强紫色成分
            r, g, b = color
            purple_colors.append((min(1.0, r * 0.5 + 0.5), min(1.0, g * 0.3 + 0.3), min(1.0, b * 0.7 + 0.3)))
        return purple_colors

    def analyze_token_importance(self, text, output_path='./token_importance.png'):
        """分析token重要性"""
        print("🔍 分析token重要性...")

        # 分词
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        # 获取原始预测
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            original_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        # 计算每个token的重要性
        importances = []
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            # 创建遮盖文本
            masked_ids = token_ids.copy()
            masked_ids[i] = self.tokenizer.mask_token_id
            masked_text = self.tokenizer.decode(masked_ids)

            # 预测遮盖后文本
            masked_inputs = self.tokenizer(masked_text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                masked_outputs = self.model(**masked_inputs)
                masked_probs = torch.nn.functional.softmax(masked_outputs.logits, dim=-1)[0]

            # 计算重要性（概率差异）
            importance = torch.abs(original_probs - masked_probs).sum().item()
            importances.append((token, importance))

        # 排序并选择前15个
        importances.sort(key=lambda x: x[1], reverse=True)
        top_tokens = importances[:15]

        # 创建美观的条形图
        fig, ax = plt.subplots(figsize=(12, 8))

        tokens_list = [t[0] for t in top_tokens]
        values = [t[1] for t in top_tokens]

        # 使用紫色渐变
        colors = self.get_purple_colors(len(tokens_list))

        bars = ax.barh(range(len(tokens_list)), values, color=colors, edgecolor='#4B0082', linewidth=0.5)

        # 美化图表
        ax.set_yticks(range(len(tokens_list)))
        ax.set_yticklabels(tokens_list, fontsize=11)
        ax.set_xlabel('重要性分数', fontsize=12, fontweight='bold')
        ax.set_title('Token重要性分析 - 紫色渐变', fontsize=16, fontweight='bold', color='#4B0082', pad=20)

        # 添加网格
        ax.grid(True, axis='x', alpha=0.3, linestyle='--', color='gray')

        # 添加数值标签
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{value:.4f}', ha='left', va='center', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # 设置背景色
        ax.set_facecolor('#FAF9F6')
        fig.patch.set_facecolor('#FAF9F6')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Token重要性图已保存: {output_path}")

        return top_tokens

    def create_prediction_heatmap(self, texts, output_path='./prediction_heatmap.png'):
        """创建预测热力图"""
        print("🔥 创建预测热力图...")

        # 计算预测概率
        all_probs = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs)

        # 创建热力图
        fig, ax = plt.subplots(figsize=(14, 8))

        # 使用紫色渐变色彩映射
        cmap = sns.color_palette("magma", as_cmap=True)

        im = ax.imshow(all_probs, aspect='auto', cmap=cmap)

        # 设置标签
        ax.set_xticks(range(4))
        ax.set_xticklabels(['相似度 0', '相似度 1', '相似度 2', '相似度 3'],
                           fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(texts)))
        ax.set_yticklabels([f'案例 {i + 1}' for i in range(len(texts))],
                           fontsize=11, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('预测概率', fontsize=12, fontweight='bold')

        # 添加数值标签
        for i in range(len(texts)):
            for j in range(4):
                text_color = 'white' if all_probs[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{all_probs[i, j]:.3f}',
                        ha='center', va='center', fontsize=10,
                        color=text_color, fontweight='bold')

        # 美化
        ax.set_title('法律案例相似度预测热力图', fontsize=18,
                     fontweight='bold', color='#4B0082', pad=20)
        ax.set_facecolor('#F5F0FF')
        fig.patch.set_facecolor('#F5F0FF')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 预测热力图已保存: {output_path}")

        return all_probs

    def create_comparison_radar(self, texts, output_path='./comparison_radar.png'):
        """创建比较雷达图"""
        print("📡 创建比较雷达图...")

        # 计算预测概率
        all_probs = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                all_probs.append(probs.cpu().numpy())

        all_probs = np.array(all_probs)

        # 创建雷达图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)

        # 角度
        angles = np.linspace(0, 2 * np.pi, len(texts), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 使用紫色系颜色
        colors = ['#9370DB', '#BA68C8', '#AB47BC', '#9C27B0', '#8E24AA']

        # 绘制每个相似度等级
        for level in range(4):
            values = all_probs[:, level].tolist()
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=3,
                    label=f'相似度等级 {level}', color=colors[level % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[level % len(colors)])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f'案例 {i + 1}' for i in range(len(texts))],
                           fontsize=12, fontweight='bold')

        # 设置径向标签
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                           fontsize=10, color='#666666')

        # 美化
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAF9F6')
        fig.patch.set_facecolor('#FAF9F6')

        plt.title('法律案例相似度预测对比雷达图', fontsize=18,
                  fontweight='bold', color='#4B0082', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                   fontsize=11, framealpha=0.9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 比较雷达图已保存: {output_path}")

        return all_probs

    def create_summary_dashboard(self, output_dir='./simple_purple_dashboard'):
        """创建综合仪表板"""
        print("=" * 60)
        print("🎨 简化美观分析 - 紫色配色方案")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # 示例文本
        texts = [
            "被告人盗窃手机价值3000元 [SEP] 盗窃罪立案标准为2000-50000元",
            "交通事故致人死亡 [SEP] 交通肇事罪需负刑事责任",
            "合同违约损失100万 [SEP] 违约金不得超过实际损失的30%",
            "贪污公款50万元 [SEP] 贪污数额巨大标准为20-300万元",
            "故意伤害致人轻伤 [SEP] 故意伤害罪需达到轻伤标准"
        ]

        # 1. Token重要性分析
        print("\n1. Token重要性分析...")
        self.analyze_token_importance(
            texts[0],
            os.path.join(output_dir, 'token_importance.png')
        )

        # 2. 预测热力图
        print("\n2. 预测热力图分析...")
        self.create_prediction_heatmap(
            texts,
            os.path.join(output_dir, 'prediction_heatmap.png')
        )

        # 3. 比较雷达图
        print("\n3. 比较雷达图分析...")
        self.create_comparison_radar(
            texts,
            os.path.join(output_dir, 'comparison_radar.png')
        )

        # 4. 创建HTML报告
        self.create_simple_html_report(output_dir)

        print(f"\n🎉 分析完成！结果保存在: {output_dir}")

    def create_simple_html_report(self, output_dir):
        """创建简单的HTML报告"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>法律模型分析报告 - 简化版</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f0ff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(148, 0, 211, 0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            color: #4B0082;
            font-size: 2em;
            background: linear-gradient(45deg, #9370DB, #4B0082);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .chart-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .chart-box {
            background: #FAF9F6;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #E6E6FA;
        }
        .chart-box h3 {
            color: #6A5ACD;
            margin-top: 0;
        }
        .chart-img {
            width: 100%;
            border-radius: 5px;
        }
        .footer {
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 法律匹配模型分析报告</h1>
            <p>简化分析版 - 紫色渐变配色</p>
        </div>

        <div class="chart-container">
            <div class="chart-box">
                <h3>🔍 Token重要性分析</h3>
                <img src="token_importance.png" class="chart-img">
                <p>显示对预测结果最重要的token及其影响程度</p>
            </div>

            <div class="chart-box">
                <h3>🔥 预测概率热力图</h3>
                <img src="prediction_heatmap.png" class="chart-img">
                <p>不同案例在各相似度等级的预测概率分布</p>
            </div>

            <div class="chart-box">
                <h3>📡 多案例比较雷达图</h3>
                <img src="comparison_radar.png" class="chart-img">
                <p>多个案例在不同相似度等级的比较分析</p>
            </div>
        </div>

        <div class="footer">
            <p>生成时间：""" + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M') + """</p>
            <p>分析方法：Token遮盖法 + 概率预测分析</p>
        </div>
    </div>
</body>
</html>
        """

        html_path = os.path.join(output_dir, 'simple_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 HTML报告已生成: {html_path}")


# ================== 主程序 ==================
if __name__ == "__main__":
    # 设置模型路径
    MODEL_PATH = "E:\\Py_Dev\\IceBerg\\lawformer_matching_model"

    # 检查路径
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
    else:
        try:
            # 创建分析器
            analyzer = SimplePurpleAnalyzer(MODEL_PATH)

            # 运行分析
            analyzer.create_summary_dashboard('./simple_purple_results')

        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback

            traceback.print_exc()
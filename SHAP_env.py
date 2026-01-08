import os
import json
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ================== 专业紫色配色方案 ==================
class PurpleColorPalette:
    """紫色系配色方案"""

    @staticmethod
    def get_purple_cmap():
        """获取紫色渐变色彩映射"""
        colors = [
            '#F3E5F5',  # 浅紫
            '#E1BEE7',  # 淡紫
            '#CE93D8',  # 紫丁香
            '#BA68C8',  # 中紫
            '#AB47BC',  # 紫水晶
            '#9C27B0',  # 紫色
            '#8E24AA',  # 紫罗兰
            '#7B1FA2',  # 深紫
            '#6A1B9A',  # 蓝紫
            '#4A148C',  # 深蓝紫
        ]
        return sns.color_palette(colors, as_cmap=True)

    @staticmethod
    def get_shap_cmap():
        """获取SHAP风格的红-蓝-紫渐变"""
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list(
            'shap_purple',
            ['#FF0000', '#FF6B6B', '#FFA8A8', '#FFFFFF', '#E6E6FA', '#9370DB', '#4B0082']
        )

    @staticmethod
    def get_sequential_purple():
        """顺序紫色调色板"""
        return sns.color_palette("magma", as_cmap=True)

    @staticmethod
    def get_bar_colors(values, cmap_name='purple'):
        """根据值获取条形图颜色"""
        if cmap_name == 'purple':
            norm_vals = (values - values.min()) / (values.max() - values.min() + 1e-8)
            cmap = plt.cm.Purples
            return [cmap(v) for v in norm_vals]
        elif cmap_name == 'red_blue':
            colors = []
            for v in values:
                if v >= 0:
                    intensity = min(abs(v) / (abs(values).max() + 1e-8), 0.8)
                    colors.append((1.0, 0.7 - 0.6 * intensity, 0.7 - 0.6 * intensity))
                else:
                    intensity = min(abs(v) / (abs(values).max() + 1e-8), 0.8)
                    colors.append((0.7 - 0.6 * intensity, 0.7 - 0.6 * intensity, 1.0))
            return colors
        elif cmap_name == 'magma':
            cmap = plt.cm.magma
            norm_vals = (values - values.min()) / (values.max() - values.min() + 1e-8)
            return [cmap(v) for v in norm_vals]


# ================== 修复的SHAP分析器 ==================
class FixedSHAPAnalyzer:
    """修复的SHAP分析器"""

    def __init__(self, model_path):
        self.model_path = model_path
        print(f"📥 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 使用设备: {self.device}")
        self.model.to(self.device)
        self.model.eval()

    def predict_proba(self, texts):
        """预测函数 - 用于SHAP解释器"""
        if isinstance(texts, str):
            texts = [texts]

        # 编码文本
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        # 预测
        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        return probs.cpu().numpy()

    def create_sample_texts(self):
        """创建示例文本"""
        samples = [
            "被告人盗窃他人手机价值5000元 [SEP] 盗窃罪是指以非法占有为目的，秘密窃取公私财物数额较大的行为。",
            "交通事故致一人死亡两人受伤 [SEP] 交通肇事罪是指违反交通运输管理法规，因而发生重大事故的行为。",
            "合同纠纷涉及违约金支付100万元 [SEP] 合同纠纷的解决需依据合同约定和相关法律规定，特别是关于违约金的部分。",
            "故意伤害致人轻伤二级 [SEP] 故意伤害罪根据伤害程度分为轻伤、重伤和致人死亡，轻伤二级属于轻伤范畴。",
            "贪污公款150万元 [SEP] 贪污罪是指国家工作人员利用职务上的便利，侵吞、窃取、骗取公共财物的行为。"
        ]
        return samples

    def analyze_with_fixed_beeswarm(self, output_path='./shap_fixed/beeswarm.png'):
        """使用修复的方法生成蜂群图"""
        print("🐝 生成蜂群图...")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 获取样本文本
        texts = self.create_sample_texts()

        try:
            # 方法1：使用正确的Explainer格式
            explainer = shap.Explainer(
                self.predict_proba,
                self.tokenizer,
                algorithm='permutation'
            )

            # 计算SHAP值
            shap_values = explainer(texts)

            # 绘制蜂群图
            plt.figure(figsize=(14, 8))
            shap.plots.beeswarm(shap_values, show=False, max_display=15)

            # 美化
            ax = plt.gca()
            ax.set_facecolor('#FAF9F6')
            ax.figure.set_facecolor('#FAF9F6')
            ax.set_title('SHAP蜂群图 - 法律特征重要性', fontsize=16,
                         fontweight='bold', color='#6A5ACD')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight',
                        facecolor='#FAF9F6', edgecolor='none')
            plt.close()

            print(f"✅ 蜂群图已保存: {output_path}")
            return shap_values

        except Exception as e:
            print(f"⚠️ 蜂群图生成失败: {e}")
            print("尝试备用方法...")
            return self._alternative_beeswarm(texts, output_path)

    def _alternative_beeswarm(self, texts, output_path):
        """备用方法：手动计算特征重要性"""
        print("使用备用方法计算特征重要性...")

        # 计算每个token的重要性
        token_importance = {}

        for text in texts:
            # 获取原始预测
            original_probs = self.predict_proba(text)[0]

            # 分词
            words = text.split()

            for i, word in enumerate(words):
                if word not in ['[SEP]', '[CLS]']:
                    # 遮盖当前词
                    masked_words = words.copy()
                    masked_words[i] = '[MASK]'
                    masked_text = ' '.join(masked_words)

                    # 预测遮盖后的文本
                    masked_probs = self.predict_proba(masked_text)[0]

                    # 计算重要性（概率变化）
                    importance = np.abs(original_probs - masked_probs).sum()

                    # 累加
                    if word in token_importance:
                        token_importance[word] += importance
                    else:
                        token_importance[word] = importance

        # 排序并选择前20个
        top_tokens = sorted(token_importance.items(), key=lambda x: x[1], reverse=True)[:20]

        # 创建条形图
        words = [t[0] for t in top_tokens]
        importances = [t[1] for t in top_tokens]

        # 使用紫色渐变
        colors = PurpleColorPalette.get_bar_colors(np.array(importances), 'purple')

        # 绘制图表
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(range(len(words)), importances, color=colors, edgecolor='#4B0082')

        # 美化
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)
        ax.set_xlabel('特征重要性', fontsize=12)
        ax.set_title('Top 20 重要法律特征（遮盖法）', fontsize=16,
                     fontweight='bold', color='#6A5ACD')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        # 添加数值标签
        for bar, value in zip(bars, importances):
            width = bar.get_width()
            ax.text(width + max(importances) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{value:.4f}', ha='left', va='center', fontsize=10)

        ax.set_facecolor('#FAF9F6')
        fig.patch.set_facecolor('#FAF9F6')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 特征重要性图已保存: {output_path}")
        return top_tokens

    def analyze_with_waterfall(self, output_dir='./shap_fixed/waterfall'):
        """生成瀑布图"""
        print("🌊 生成瀑布图...")

        os.makedirs(output_dir, exist_ok=True)

        texts = self.create_sample_texts()

        for i, text in enumerate(texts[:3]):  # 只分析前3个样本
            try:
                # 使用Permutation解释器
                explainer = shap.Explainer(
                    self.predict_proba,
                    masker=self.tokenizer,
                    algorithm='permutation'
                )

                # 计算SHAP值
                shap_values = explainer([text])

                # 绘制瀑布图
                plt.figure(figsize=(12, 6))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)

                # 美化
                ax = plt.gca()
                ax.set_title(f'瀑布图 - 样本 {i + 1}', fontsize=14,
                             fontweight='bold', color='#6A5ACD')
                ax.set_facecolor('#FAF9F6')
                ax.figure.set_facecolor('#FAF9F6')

                # 保存
                output_path = os.path.join(output_dir, f'waterfall_{i + 1}.png')
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight',
                            facecolor='#FAF9F6', edgecolor='none')
                plt.close()

                print(f"  ✅ 瀑布图 {i + 1}: {output_path}")

            except Exception as e:
                print(f"  ⚠️ 瀑布图 {i + 1} 生成失败: {e}")
                # 尝试生成简单的条形图替代
                self._create_simple_waterfall(text, i, output_dir)

    def _create_simple_waterfall(self, text, idx, output_dir):
        """创建简单的瀑布图替代"""
        # 分词
        words = text.split()[:15]  # 只取前15个词

        # 计算每个词的重要性
        original_probs = self.predict_proba(text)[0]
        importances = []

        for i, word in enumerate(words):
            if word not in ['[SEP]', '[CLS]']:
                # 遮盖
                masked_words = text.split()
                masked_words[i] = '[MASK]'
                masked_text = ' '.join(masked_words)
                masked_probs = self.predict_proba(masked_text)[0]

                # 重要性
                importance = (original_probs - masked_probs).sum()
                importances.append(importance)
            else:
                importances.append(0)

        # 绘制
        fig, ax = plt.subplots(figsize=(12, 6))

        # 计算累计值
        cumulative = 0
        for i, (word, imp) in enumerate(zip(words, importances)):
            if i == 0:
                ax.bar(i, imp, color='#E6E6FA', edgecolor='#4B0082')
            else:
                ax.bar(i, imp, bottom=cumulative, color='#9370DB' if imp > 0 else '#FF6B6B',
                       edgecolor='#4B0082')
            cumulative += imp

        # 添加连接线
        for i in range(len(words) - 1):
            x1, x2 = i, i + 1
            y1 = sum(importances[:i + 1])
            y2 = sum(importances[:i + 2])
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)

        # 美化
        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('累计SHAP值', fontsize=12)
        ax.set_title(f'瀑布图 - 样本 {idx + 1} (简化版)', fontsize=14,
                     fontweight='bold', color='#6A5ACD')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        ax.set_facecolor('#FAF9F6')
        fig.patch.set_facecolor('#FAF9F6')

        output_path = os.path.join(output_dir, f'waterfall_{idx + 1}_simple.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 简化瀑布图 {idx + 1}: {output_path}")

    def create_heatmap_analysis(self, output_path='./shap_fixed/heatmap.png'):
        """创建热力图分析"""
        print("🔥 创建热力图...")

        texts = self.create_sample_texts()

        # 计算每个样本的预测概率
        all_probs = []
        for text in texts:
            probs = self.predict_proba(text)[0]
            all_probs.append(probs)

        all_probs = np.array(all_probs)

        # 创建热力图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 使用红-蓝-紫渐变
        cmap = PurpleColorPalette.get_shap_cmap()

        im = ax.imshow(all_probs, aspect='auto', cmap=cmap)

        # 设置标签
        ax.set_xticks(range(4))
        ax.set_xticklabels(['相似度0', '相似度1', '相似度2', '相似度3'], fontsize=12)
        ax.set_yticks(range(len(texts)))
        ax.set_yticklabels([f'案例{i + 1}' for i in range(len(texts))], fontsize=11)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('预测概率', fontsize=12)

        # 添加数值标签
        for i in range(len(texts)):
            for j in range(4):
                ax.text(j, i, f'{all_probs[i, j]:.2f}',
                        ha='center', va='center', fontsize=10,
                        color='white' if all_probs[i, j] > 0.5 else 'black')

        ax.set_title('预测概率热力图', fontsize=16,
                     fontweight='bold', color='#6A5ACD', pad=20)
        ax.set_facecolor('#FAF9F6')
        fig.patch.set_facecolor('#FAF9F6')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 热力图已保存: {output_path}")

    def create_radar_chart(self, output_path='./shap_fixed/radar.png'):
        """创建雷达图"""
        print("📡 创建雷达图...")

        texts = self.create_sample_texts()

        # 计算每个样本的预测概率
        all_probs = []
        categories = ['盗窃案', '交通肇事', '合同纠纷', '故意伤害', '贪污案']

        for text in texts:
            probs = self.predict_proba(text)[0]
            all_probs.append(probs)

        all_probs = np.array(all_probs)

        # 创建雷达图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        # 角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        # 绘制每个相似度等级
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        for level in range(4):
            values = all_probs[:, level].tolist()
            values += values[:1]  # 闭合

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=f'相似度{level}', color=colors[level])
            ax.fill(angles, values, alpha=0.1, color=colors[level])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)

        # 美化
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, color='gray')

        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#FAF9F6')
        fig.patch.set_facecolor('#FAF9F6')

        plt.title('法律案例相似度预测雷达图', fontsize=16,
                  fontweight='bold', color='#6A5ACD', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 雷达图已保存: {output_path}")

    def run_full_analysis(self, output_dir='./shap_fixed_results'):
        """运行完整分析"""
        print("=" * 60)
        print("🎨 美观的SHAP可解释性分析 - 修复版")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)

        # 1. 蜂群图
        print("\n1. 生成蜂群图...")
        self.analyze_with_fixed_beeswarm(os.path.join(output_dir, 'beeswarm.png'))

        # 2. 瀑布图
        print("\n2. 生成瀑布图...")
        self.analyze_with_waterfall(os.path.join(output_dir, 'waterfall'))

        # 3. 热力图
        print("\n3. 生成热力图...")
        self.create_heatmap_analysis(os.path.join(output_dir, 'heatmap.png'))

        # 4. 雷达图
        print("\n4. 生成雷达图...")
        self.create_radar_chart(os.path.join(output_dir, 'radar.png'))

        # 5. 创建HTML报告
        self.create_html_report(output_dir)

        print(f"\n🎉 分析完成！结果保存在: {output_dir}")

    def create_html_report(self, output_dir):
        """创建HTML报告"""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>法律模型SHAP分析报告 - 修复版</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #9370DB;
        }
        .header h1 {
            color: #4B0082;
            font-size: 2.2em;
            margin-bottom: 10px;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 25px;
            margin-bottom: 40px;
        }
        .chart-card {
            background: #FAF9F6;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .chart-card h3 {
            color: #4B0082;
            margin-top: 0;
            border-left: 4px solid #9370DB;
            padding-left: 10px;
        }
        .chart-image {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #E6E6FA;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #E6E6FA;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 法律匹配模型SHAP分析报告</h1>
            <p>模型可解释性分析 - 紫色渐变配色方案</p>
        </div>

        <div class="chart-grid">
            <div class="chart-card">
                <h3>🐝 蜂群图 - 特征分布</h3>
                <img src="beeswarm.png" alt="蜂群图" class="chart-image">
                <p>展示特征重要性分布，颜色表示特征值</p>
            </div>

            <div class="chart-card">
                <h3>🔥 热力图 - 预测概率</h3>
                <img src="heatmap.png" alt="热力图" class="chart-image">
                <p>不同案例的相似度预测概率分布</p>
            </div>

            <div class="chart-card">
                <h3>📡 雷达图 - 多维分析</h3>
                <img src="radar.png" alt="雷达图" class="chart-image">
                <p>各案例在不同相似度等级的预测分布</p>
            </div>

            <div class="chart-card">
                <h3>🌊 瀑布图分析</h3>
                <p>单个样本的特征贡献分解：</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
                    <img src="waterfall/waterfall_1.png" alt="瀑布图1" style="width: 100%; border-radius: 5px;">
                    <img src="waterfall/waterfall_2.png" alt="瀑布图2" style="width: 100%; border-radius: 5px;">
                </div>
            </div>
        </div>

        <div class="footer">
            <p>📅 报告生成时间：""" + pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S') + """</p>
            <p>🎨 配色方案：红-蓝-紫渐变 | SHAP可解释性分析</p>
        </div>
    </div>
</body>
</html>
        """

        # 保存HTML文件
        html_path = os.path.join(output_dir, 'shap_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📄 HTML报告已生成: {html_path}")


# ================== 主程序 ==================
def main():
    # 设置模型路径
    MODEL_PATH = "E:\\Py_Dev\\IceBerg\\lawformer_matching_model"

    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        return

    try:
        # 创建分析器
        analyzer = FixedSHAPAnalyzer(MODEL_PATH)

        # 运行完整分析
        analyzer.run_full_analysis('./shap_fixed_results')

    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
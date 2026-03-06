import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import font_manager


# 中文字体设置
font_path = "C:/Windows/Fonts/simhei.ttf"
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 保存路径设置
result_dir = "../result"
os.makedirs(result_dir, exist_ok=True)

def save_fig(fig, filename):
    """保存图像并关闭 figure"""
    path = os.path.join(result_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f">图像已保存: {path}")


# 加载数据
results_csv = os.path.join(result_dir, "model_results.csv")
y_test_csv = os.path.join(result_dir, "y_test.csv")

if not os.path.exists(results_csv) or not os.path.exists(y_test_csv):
    raise FileNotFoundError("请先确保 model_train.py 已生成 model_results.csv 和 y_test.csv 文件")

results_df = pd.read_csv(results_csv)
y_test = pd.read_csv(y_test_csv)['target']


# 绘制性能指标柱状图
metrics = ["准确率", "精确率", "召回率", "F1值"]

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="模型", y=metric, hue="模型", data=results_df, dodge=False, palette="Set2", legend=False)
    ax.set_title(f"不同模型{metric}对比", fontproperties=my_font)
    ax.tick_params(axis='x', rotation=20)  # 代替 set_xticklabels
    plt.tight_layout()
    save_fig(fig, f"model_{metric}.png")


# 绘制每个模型的混淆矩阵
for idx, row in results_df.iterrows():
    model_name = row["模型"]
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    y_pred_csv = os.path.join(result_dir, f"y_pred_{safe_name}.csv")

    if not os.path.exists(y_pred_csv):
        print(f"【ERROR】未找到预测文件: {y_pred_csv}，跳过此模型")
        continue

    y_pred = pd.read_csv(y_pred_csv)['target']
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["恶性(0)", "良性(1)"],
                yticklabels=["恶性(0)", "良性(1)"], ax=ax)
    ax.set_xlabel("预测值", fontproperties=my_font)
    ax.set_ylabel("真实值", fontproperties=my_font)
    ax.set_title(f"{model_name} 混淆矩阵", fontproperties=my_font, fontsize=12)
    plt.tight_layout()
    save_fig(fig, f"{safe_name}_confusion_matrix.png")


# 输出最佳模型
best_model_name = results_df.sort_values(by="准确率", ascending=False).iloc[0]["模型"]
print(f"\n>>最佳模型: {best_model_name}")

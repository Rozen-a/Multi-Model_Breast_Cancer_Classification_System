import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from matplotlib import font_manager

# 中文字体设置
font_path = "C:/Windows/Fonts/simhei.ttf"
my_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 设置保存路径
save_dir = "../data"
os.makedirs(save_dir, exist_ok=True)

def save_fig(filename):
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f">图像已保存: {path}")

# 数据加载
data_path = "../data/breast_cancer.csv"
if not os.path.exists(data_path):
    print(f"错误：文件不存在 - {data_path}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"尝试的路径: {os.path.abspath(data_path)}")
df = pd.read_csv(data_path)

# 良性/恶性 类别样本分布
plt.figure(figsize=(6, 4))
sns.countplot(x='target', hue='target', data=df, palette='Set2', dodge=False, legend=False)
plt.title("良性 / 恶性 样本分布", fontproperties=my_font)
plt.xlabel("类别（0 = 恶性，1 = 良性）", fontproperties=my_font)
plt.ylabel("样本数量", fontproperties=my_font)
plt.tick_params(axis='x')
plt.tight_layout()
save_fig("class_distribution.png")

# 特征相关性热力图
plt.figure(figsize=(12, 10))
corr = df.drop(columns=['target']).corr()
sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("特征相关性热力图", fontproperties=my_font)
plt.tight_layout()
save_fig("correlation_heatmap.png")

# PCA 降维可视化
X = df.drop(columns=['target'])
y = df['target']
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca['target'] = y

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='target', data=df_pca, palette='Set1', alpha=0.7)
plt.title("PCA 降维可视化", fontproperties=my_font)
plt.tight_layout()
save_fig("pca_visualization.png")

# 单特征分布（平均半径）
plt.figure(figsize=(6, 4))
sns.histplot(df['mean radius'], kde=True, color='teal')
plt.title("平均半径分布", fontproperties=my_font)
plt.tight_layout()
save_fig("mean_radius_distribution.png")

print("\n==所有可视化图像已生成并保存到 data 文件夹==")

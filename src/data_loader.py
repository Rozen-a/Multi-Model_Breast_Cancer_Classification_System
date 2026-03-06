import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 设置图形风格（可选）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")


def load_data(test_size=0.2, random_state=42, save_csv=True):
    """加载乳腺癌数据集并进行预处理"""
    # 加载数据集
    breast_cancer = load_breast_cancer()

    # 转换为DataFrame
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = pd.Series(breast_cancer.target, name="target")

    # 保存CSV（可选）
    if save_csv:
        df = X.copy()
        df["target"] = y
        df.to_csv("../data/breast_cancer.csv", index=False)
        print("✅ 数据集已保存到 data/breast_cancer.csv")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, breast_cancer


if __name__ == "__main__":
    # 测试模块是否正常工作
    X_train, X_test, y_train, y_test, data_info = load_data()
    print("📊 特征数量:", X_train.shape[1])
    print("🔍 类别分布：", np.bincount(y_train))
    print("📖 数据集描述：\n")
    print(data_info.DESCR)

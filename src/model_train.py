import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# 路径设置
data_path = "../data/breast_cancer.csv"
model_dir = "../model"
result_dir = "../result"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# 加载数据集
df = pd.read_csv(data_path)
X = df.drop(columns=['target'])
y = df['target']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 模型定义
models = {
    "逻辑回归 (Logistic Regression)": LogisticRegression(max_iter=500),
    "K近邻 (KNN)": KNeighborsClassifier(n_neighbors=5),
    "决策树 (Decision Tree)": DecisionTreeClassifier(random_state=42),
    "随机森林 (Random Forest)": RandomForestClassifier(n_estimators=100, random_state=42),
    "支持向量机 (SVM)": SVC(kernel='rbf', probability=True, random_state=42)
}

# 训练模型、保存模型与预测结果
results = []

for name, model in models.items():
    # 模型训练
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([name, acc, pre, rec, f1])

    print(f"模型: {name}")
    print(classification_report(y_test, y_pred, target_names=["恶性(0)", "良性(1)"]))

    # 保存训练好的模型
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    model_path = os.path.join(model_dir, f"{safe_name}.joblib")
    joblib.dump(model, model_path)

    # 保存预测结果
    y_pred_df = pd.DataFrame({'target': y_pred})
    y_pred_path = os.path.join(result_dir, f"y_pred_{safe_name}.csv")
    y_pred_df.to_csv(y_pred_path, index=False)
    print(f">模型已保存: {model_path}")
    print(f">预测结果已保存: {y_pred_path}")
    print("\n" + "-" * 60)

# 保存 y_test
y_test_df = pd.DataFrame({'target': y_test})
y_test_df.to_csv(os.path.join(result_dir, "y_test.csv"), index=False)

# 保存汇总指标
results_df = pd.DataFrame(results, columns=["模型", "准确率", "精确率", "召回率", "F1值"])
results_df.to_csv(os.path.join(result_dir, "model_results.csv"), index=False)
print(f"\n>>各模型性能对比已保存到: {os.path.join(result_dir, 'model_results.csv')}")

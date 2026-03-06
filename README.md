# 多模型乳腺癌良恶性智能诊断系统

基于多种经典机器学习模型（逻辑回归、KNN、决策树、随机森林、SVM）的乳腺癌良恶性智能诊断与比较分析系统。项目基于 UCI 的 Wisconsin Breast Cancer Diagnostic（WDBC）公开数据集，并通过 Scikit‑Learn 提供的 `load_breast_cancer` 接口加载数据，对多模型进行训练、评估与可视化展示，方便教学演示和实验对比。

本项目将乳腺肿块的良/恶性判定建模为一个**二分类问题**，重点比较多种经典机器学习算法在同一数据集上的性能差异、误判特性及适用场景：一方面为教学与课程设计提供一个结构清晰、可复现的实验平台，另一方面为后续在真实临床数据上引入更复杂模型（如深度学习、集成学习）提供基线与参考。

---

## 一、项目功能概述

### 1. 项目目标与特色（对应报告的“项目背景 / 研究范围 / 项目特色”）

- **项目目标**：在公开的 WDBC 乳腺癌数据集上，构建一个端到端的智能诊断实验平台，完成从数据预处理、探索性分析、多模型训练到模型评估与可视化的完整流程。
- **研究重点**：比较逻辑回归、支持向量机、KNN、决策树和随机森林在同一数据集上的表现，分析不同模型的**误判模式**与**适用场景**，尤其关注恶性样本的召回能力（减少漏诊）。
- **项目特色**：
  - 强调**多模型对比**而非单一模型结果；
  - 将统计分析、可视化与分类结果结合，形成一个结构化的小型实验框架；
  - 代码结构清晰，便于学生和研究者在此基础上扩展更多模型或指标。

### 2. 功能概览（对应报告的“功能需求”）

- **数据加载与预处理**
  - 调用 `sklearn.datasets.load_breast_cancer` 加载乳腺癌数据集；
  - 使用 `StandardScaler` 对特征进行标准化；
  - 划分训练集与测试集，并可将完整数据保存为 `data/breast_cancer.csv`。

- **数据可视化分析（探索性数据分析 EDA）**
  - 良性 / 恶性样本数量分布；
  - 特征相关性热力图；
  - PCA 降维后的二维可视化；
  - 典型特征（如 mean radius）的分布图。

- **多模型训练与预测**
  - 逻辑回归（Logistic Regression）
  - K 近邻（KNN）
  - 决策树（Decision Tree）
  - 随机森林（Random Forest）
  - 支持向量机（SVM）
  - 自动保存每个模型的训练结果与预测结果。

- **模型性能评估与对比**
  - 计算并保存每个模型的主要评估指标：
    - **准确率（Accuracy）**：整体分类正确的比例；
    - **精确率（Precision）**：预测为某类的样本中，真正属于该类的比例；
    - **召回率（Recall）**：所有真实属于某类的样本中，被模型正确预测的比例；
    - **F1 值（F1-score）**：精确率与召回率的调和平均，用于综合权衡；
  - 生成各模型性能指标的对比柱状图；
  - 为每个模型绘制**混淆矩阵**，用于分析 TP/FN/FP/TN 及误判类型；
  - 自动识别并打印当前表现最优的模型。

- **一键运行全流程**
  - 通过 `main.py` 按顺序完成：
    1. 数据可视化；
    2. 模型训练；
    3. 模型评估与可视化。

---

## 二、项目目录结构

```text
Multi-Model_Breast_Cancer_Classification_System
├── main.py                       # 一键执行入口脚本
├── check_all..py                 # 辅助脚本：检查 src 下脚本结构
├── requirements.txt              # Python 依赖列表
├── 基于多机器学习模型的乳腺癌良恶性智能诊断系统项目报告.docx    # 项目论文报告
├── src
│   ├── data_loader.py            # 从 sklearn 加载数据并预处理（可选使用）
│   ├── visualize.py              # 生成各类数据分布与降维可视化图
│   ├── model_train.py            # 多模型训练与结果保存
│   └── model_evaluate.py         # 各模型指标对比与混淆矩阵绘制
├── data
│   ├── breast_cancer.csv         # 乳腺癌数据集（特征 + target）
│   ├── class_distribution.png    # 类别分布图
│   ├── correlation_heatmap.png   # 特征相关性热力图
│   ├── mean_radius_distribution.png
│   └── pca_visualization.png     # PCA 降维可视化结果
├── model
│   ├── 逻辑回归_Logistic_Regression.joblib
│   ├── K近邻_KNN.joblib
│   ├── 决策树_Decision_Tree.joblib
│   ├── 随机森林_Random_Forest.joblib
│   └── 支持向量机_SVM.joblib
└── result
    ├── model_results.csv         # 各模型综合指标对比表
    ├── y_test.csv                # 测试集真实标签
    ├── y_pred_*.csv              # 各模型在测试集上的预测结果
    ├── model_准确率.png
    ├── model_精确率.png
    ├── model_召回率.png
    ├── model_F1值.png
    ├── model_accuracy_comparison.png (若有)
    ├── *confusion_matrix.png     # 各模型混淆矩阵可视化
    └── ...
```

> 说明：部分图片或文件在首次运行相关脚本后自动生成。

---

## 三、环境依赖

- **Python**：建议 3.8 及以上版本
- **依赖库**（已在 `requirements.txt` 中列出，可按需精简/扩展）：
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib
  - torch（当前主流程未使用，可用于后续扩展为深度学习模型）

### 安装依赖

在项目根目录下执行：

```bash
pip install -r requirements.txt
```

> Windows 环境下建议使用虚拟环境（例如 `venv` 或 Anaconda）。

---

## 四、中文字体/绘图说明

本项目大量图表标题、坐标轴等使用中文显示：

- `src/visualize.py` 与 `src/model_evaluate.py` 中均通过：
  - `plt.rcParams['font.sans-serif'] = ['SimHei']`
  - 使用 `C:/Windows/Fonts/simhei.ttf` 作为字体路径
- **请确保系统中存在 `simhei.ttf`（黑体）字体**，否则 Matplotlib 可能无法正常显示中文或报错。
  - 若无此字体，可安装「微软雅黑 / 黑体」等中文字体，或在代码中修改为本机存在的字体路径。

---

## 五、数据说明（对应报告的“问题定义 / 数据集选用 / 数据划分与预处理”）

- **问题类型**：二分类问题——预测乳腺肿块为**恶性（0）**还是**良性（1）**；
- **输入特征**：每个样本包含 30 个连续数值特征，来源于乳腺肿块细针穿刺（FNA）图像的统计量（如 mean radius、mean texture 等）；
- **输出标签 `target` 含义**：
  - `0` 表示恶性（Malignant）
  - `1` 表示良性（Benign）
- **数据来源**：`sklearn.datasets.load_breast_cancer` 内置的 Wisconsin Breast Cancer Diagnostic（WDBC）数据集，对应 UCI 公开数据集 *Breast Cancer Wisconsin (Diagnostic)*（可参考 `https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic`）；
- **数据规模与分布**：
  - 样本总数：569；
  - 特征数：30 个连续数值特征；
  - 类别分布：良性样本约 357 条，恶性样本约 212 条，存在一定但不极端的类别不平衡。

在当前项目中：

- `data/breast_cancer.csv` 为从 WDBC 数据集导出并整理好的结构化数据文件（30 个原始数值特征 + 1 列 `target` 标签），特征标准化在训练阶段由 `StandardScaler` 完成；
- 训练/测试集划分采用 **80% / 20%** 随机划分，并使用固定随机种子 `random_state=42` 与**分层抽样（`stratify=y`）**，以保证结果可复现且训练/测试集中类别比例接近；
- 如需重新生成 `breast_cancer.csv` 文件，可运行 `src/data_loader.py`，脚本会再次从 sklearn 加载数据并保存到 `data` 目录。

---

## 六、运行说明

### 1. 克隆项目 / 拷贝代码

将项目代码放置到本地，例如：

```text
D:\rozen_file\Program\Multi-Model_Breast_Cancer_Classification_System
```

### 2. 创建与激活虚拟环境（可选但推荐）

```bash
cd D:\rozen_file\Program\Multi-Model_Breast_Cancer_Classification_System

python -m venv venv
venv\Scripts\activate  # Windows PowerShell / CMD
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 一键运行完整流程

在项目根目录执行：

```bash
python main.py
```

程序将依次执行：

1. **数据可视化（`src/visualize.py`）**
   - 生成并保存数据分布、特征相关性、PCA 降维等图像到 `data` 目录；
2. **模型训练（`src/model_train.py`）**
   - 训练五种机器学习模型；
   - 将训练好的模型以 `.joblib` 格式保存到 `model` 目录；
   - 将各模型在测试集上的预测结果保存为 `result/y_pred_*.csv`；
   - 将测试集真实标签保存为 `result/y_test.csv`；
   - 输出并保存 `result/model_results.csv`。
3. **模型评估与可视化（`src/model_evaluate.py`）**
   - 从 `model_results.csv` 与各 `y_pred_*.csv` 中读取结果；
   - 绘制各模型的准确率、精确率、召回率、F1 值柱状对比图；
   - 绘制各模型的混淆矩阵并保存到 `result` 目录；
   - 在控制台输出当前表现最优的模型名称。

> 若 `result/model_results.csv` 或 `result/y_test.csv` 不存在，`model_evaluate.py` 会报错提示需先运行 `model_train.py`。

---

## 七、主要脚本说明

- **`main.py`**
  - 项目入口脚本，按顺序执行可视化、训练与评估；
  - 适合「一键跑通」整个流程。

- **`src/visualize.py`**
  - 使用 `data/breast_cancer.csv` 进行 EDA 和多种图表绘制；
  - 输出图像保存在 `data` 目录。

- **`src/model_train.py`**
  - 读取 `data/breast_cancer.csv`；
  - 标准化特征、划分训练/测试集（分层抽样）；
  - 训练多种模型并计算多种指标；
  - 将模型、预测结果和综合指标保存到 `model` 与 `result` 目录。

- **`src/model_evaluate.py`**
  - 读取 `model_results.csv` 与预测结果；
  - 绘制性能柱状对比图与混淆矩阵；
  - 自动输出当前最佳模型。

- **`src/data_loader.py`（可选工具脚本）**
  - 从 sklearn 原始加载乳腺癌数据集；
  - 进行标准化与训练/测试划分测试；
  - 可用于单独调试数据加载流程或重新生成 `breast_cancer.csv`。

- **`check_all..py`**
  - 用于检查 `src` 目录下主要脚本的行数、函数定义和主入口等信息，便于快速浏览代码结构。

---

## 八、结果查看与扩展方向

- 在 **`data` 目录** 可以查看：
  - 类别分布图、特征相关性热力图、PCA 可视化、典型特征分布等图像；
- 在 **`result` 目录** 可以查看：
  - `model_results.csv`：各模型的准确率、精确率、召回率、F1 值；
  - 各 `model_*.png`：性能指标的对比图；
  - 各 `*_confusion_matrix.png`：模型混淆矩阵图；
  - `y_test.csv` 和 `y_pred_*.csv`：用于进一步分析或绘制 ROC 曲线、PR 曲线等。

### 后续可扩展方向（建议）

- 引入更多模型（如 XGBoost、LightGBM、神经网络等）；
- 增加 ROC 曲线、AUC 指标、多阈值分析等；
- 将脚本封装为简单 Web 界面或可交互 Notebook；
- 使用 `torch` 引入深度学习模型与传统机器学习模型进行对比。

---

如需在此基础上继续扩展功能（如加入前端界面、API 服务、更多模型或特征工程流程），可以在 `src` 目录中新建模块，并在 `main.py` 中增加对应的调用入口。

---

## 九、实验结果概览（与报告对应）

在 Wisconsin Breast Cancer Diagnostic 数据集上的实验结果与报告内容保持一致，主要结论如下（具体数值可在 `result/model_results.csv` 和各混淆矩阵图中查看）：

- **总体表现**：逻辑回归（Logistic Regression）与支持向量机（SVM）表现最优，测试集准确率约为 0.98，对恶性样本的召回率接近 0.98，仅各漏诊 1 例；KNN 与随机森林次之，决策树相对较差；
- **恶性样本召回**：逻辑回归与 SVM 对恶性（0 类）召回率最高，更符合临床“尽量减少漏诊”的需求；
- **误判特性**：
  - KNN 与随机森林在恶性召回上略低于 LR/SVM，但对恶性预测时的精确率较高，可在“减少误报”场景下考虑；
  - 决策树出现较多将良性误判为恶性的情况，导致恶性类精确率明显下降，体现出单棵树对边界样本和噪声的敏感性；
- **综合建议**：在本数据集上，推荐以逻辑回归或 SVM 作为首选的乳腺癌良恶性辅助诊断模型，其他模型可作为对比或集成基学习器。

---

## 十、参考文献（与报告保持一致，供进一步阅读）

[1] 汤振伟. 基于小样本学习的乳腺癌智能诊断研究[D]. 杭州电子科技大学, 2024.  
[2] 胡婷. 面向乳腺癌智能诊疗的神经网络方法研究[D]. 四川大学, 2023.  
[3] 郑惠中. 基于深度神经网络的数字乳腺断层影像病灶智能检测与诊断研究[D]. 杭州电子科技大学, 2021.  
[4] 王路. 乳腺癌病理图像的自动分类与辅助诊断系统设计[D]. 中南民族大学, 2023.  
[5] Seyma Aymaz, Samet Aymaz. A novel approach for enhanced early breast cancer detection[J]. Computer Methods in Biomechanics and Biomedical Engineering, 2025, 1–25.  
[6] Reshan A S M, Amin S, Zeb A M, et al. Advanced breast cancer prediction using Deep Neural Networks integrated with ensemble models[J]. Chemometrics and Intelligent Laboratory Systems, 2025, 262:105399-105399.  
[7] Mamun A A, Bhuiyan T, Hassan M M, et al. Exploring the Best Machine Learning Models for Breast Cancer Prediction in Wisconsin[J]. International Journal of Advanced Computer Science and Applications (IJACSA), 2025, 16(1).  
[8] Arnob B K A, Jony I A. Comparing Machine Learning Algorithms for Breast Cancer Diagnosis: Wisconsin Diagnostic Dataset Analysis[J]. International Journal of Data Science and Big Data Analytics, 2024, 4(2):1–11. DOI:10.51483/IJDSBDA.4.2.2024.1-11.
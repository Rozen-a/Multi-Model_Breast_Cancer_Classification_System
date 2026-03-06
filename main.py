# main.py
import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

if __name__ == "__main__":
    print("=" * 70)
    print("多模型乳腺癌分类系统")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("\n1. 数据可视化\n")
    import src.visualize        # 直接导入，自动执行模块顶层的所有代码

    print("\n" + "=" * 70)
    print("\n2. 模型训练\n")
    import src.model_train

    print("\n" + "=" * 70)
    print("\n3. 模型评估\n")
    import src.model_evaluate

    print("\n" + "=" * 70)
    print("\n所有任务执行完成\n")
    print("=" * 70)
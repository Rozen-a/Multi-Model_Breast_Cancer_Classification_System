# check_all.py
import os


def check_file(filename):
    """检查文件内容和函数定义"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(project_root, "src", filename)

    print(f"\n{'=' * 60}")
    print(f"检查文件: {filename}")
    print(f"{'=' * 60}")

    if not os.path.exists(filepath):
        print("文件不存在!")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        # 打印文件大小
        print(f"文件大小: {len(lines)} 行")

        # 查找函数定义
        import re
        functions = []
        for i, line in enumerate(lines[:50]):  # 只检查前50行
            match = re.match(r'^\s*def\s+(\w+)\s*\(', line)
            if match:
                functions.append(match.group(1))
                print(f"  第{i + 1:3d}行: def {match.group(1)}()")

        if functions:
            print(f"\n找到的函数: {functions}")
        else:
            print("\n未找到函数定义（可能是直接执行的脚本）")

        # 显示文件开头的关键代码
        print(f"\n文件开头内容:")
        for i in range(min(15, len(lines))):
            print(f"{i + 1:3d}: {lines[i].rstrip()}")

        # 检查是否有 if __name__ == "__main__" 块
        for i, line in enumerate(lines):
            if '__main__' in line:
                print(f"\n找到 __main__ 块在第 {i + 1} 行")
                break


if __name__ == "__main__":
    print("检查项目文件结构...")
    files_to_check = ["visualize.py", "model_train.py", "model_evaluate.py"]
    for file in files_to_check:
        check_file(file)
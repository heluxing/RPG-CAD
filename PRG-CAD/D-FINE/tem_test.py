import os
from collections import defaultdict

def count_yolo_labels(root_dir):
    class_counts = defaultdict(int)  # 存储类别ID及其出现次数

    # 递归遍历所有子文件夹
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:  # YOLO格式: class_id x_center y_center width height
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return class_counts

# 示例用法
root_directory = "/home/HELUXING/Brain_project/data/sequence_dataset_len6_cls8/labels/train_6"
counts = count_yolo_labels(root_directory)

# 打印结果（按类别ID排序）
print("类别统计结果（类别ID: 边界框数量）:")
for class_id in sorted(counts.keys()):
    print(f"类别 {class_id}: {counts[class_id]} 个边界框")
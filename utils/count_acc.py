import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import argparse


def read_yolo_annotations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        classes = [line.split()[0] for line in lines]
        return set(classes)


def count_annotations_in_folders(folder1, folder2):
    folder1_files = os.listdir(folder1)
    folder2_files = os.listdir(folder2)

    common_filenames = list(set(folder1_files) & set(folder2_files))
    matching_count = 0

    for filename in common_filenames:
        file_path1 = os.path.join(folder1, filename)
        file_path2 = os.path.join(folder2, filename)

        classes1 = read_yolo_annotations(file_path1)
        classes2 = read_yolo_annotations(file_path2)

        if classes1 == classes2:
            matching_count += 1

    return matching_count


def compare_annotations_in_folders(folder1, folder2):
    folder1_files = os.listdir(folder1)
    folder2_files = os.listdir(folder2)

    common_filenames = list(set(folder1_files) & set(folder2_files))
    true_labels = []
    predicted_labels = []

    for filename in common_filenames:
        file_path1 = os.path.join(folder1, filename)
        file_path2 = os.path.join(folder2, filename)

        true_classes = read_yolo_annotations(file_path2)
        predicted_classes = read_yolo_annotations(file_path1)

        true_labels.extend(list(true_classes))
        predicted_labels.extend(list(predicted_classes))

    return true_labels, predicted_labels


# 设置文件夹路径
parser = argparse.ArgumentParser(description='Compare YOLO annotation files in two folders.')
parser.add_argument('--folder_num', help='the exp number')
args = parser.parse_args()

folder_path1 = "runs/detect/exp" + args.folder_num + "/labels"
folder_path2 = "" # true labels path

# 获取真实标签和预测标签
true_labels, predicted_labels = compare_annotations_in_folders(folder_path1, folder_path2)
matching_count = count_annotations_in_folders(folder_path1, folder_path2)
print(f"Number of annotations with matching classes: {matching_count}    acc=" + str(matching_count / 914))
# 生成混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print(conf_matrix)
# # 使用Seaborn绘制热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(true_labels)),
#             yticklabels=sorted(set(true_labels)))
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.show()
precision_per_class = precision_score(true_labels, predicted_labels, average=None)
recall_per_class = recall_score(true_labels, predicted_labels, average=None)
# 打印每个类别的精确度和召回率
for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
    print(f"Class {i + 1}: Precision = {precision:.4f}, Recall = {recall:.4f}")

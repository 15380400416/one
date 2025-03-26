import datasets as Dataset
import os
from sklearn.model_selection import train_test_split

if not os.path.exists(r"E:\Datasets\valid_data.csv"):
    raise FileNotFoundError("文件路径错误或文件不存在：E:\Datasets\valid_data.csv")

questions = []
answers = []
with open(r"E:\Datasets\valid_data.csv", 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            questions.append(parts[0])
            answers.append(parts[1])

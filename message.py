import datasets
import os
from sklearn.model_selection import train_test_split
import csv
import re

if not os.path.exists(r"E:\Datasets\valid_data.csv"):
    raise FileNotFoundError("文件路径错误或文件不存在：E:\\Datasets\\valid_data.csv")

questions = []
answers = []
with open(r"E:\Datasets\valid_data.csv", 'r', encoding='utf-8') as f:
    # 使用 DictReader 按列名读取数据
    reader = csv.DictReader(f, delimiter=',')  # 假设分隔符是逗号
    for row in reader:
        if 'problem' in row and 'answer' in row:
            questions.append(row['problem'])
            answers.append(row['answer'])

if len(questions) == 0 or len(answers) == 0:
    raise ValueError("数据集为空，请检查文件内容和格式")

# 先划分出测试集
questions_train_val, questions_test, answers_train_val, answers_test = train_test_split(
    questions, answers, test_size=0.1, random_state=42)
# 再从训练验证集中划分出训练集和验证集
questions_train, questions_val, answers_train, answers_val = train_test_split(
    questions_train_val, answers_train_val, test_size=0.111, random_state=42)

# 使用 datasets.Dataset.from_dict 创建数据集
train_dataset = datasets.Dataset.from_dict({
    'question': questions_train,
    'answear': answers_train
})
val_dataset = datasets.Dataset.from_dict({
    'question': questions_val,
    'answear': answers_val
})
test_dataset = datasets.Dataset.from_dict({
    'question': questions_test,
    'answear': answers_test
})

# 显示训练集、验证集和测试集的前五行数据
print("训练集前五行数据：")
print(train_dataset.select(range(5)))
print("\n验证集前五行数据：")
print(val_dataset.select(range(5)))
print("\n测试集前五行数据：")
print(test_dataset.select(range(5)))



def clean_text(text):
    text = re.sub('[^a-zA-Z0-9\s\.\,\?\!\(\)\[\]\{\}\-\_=\+\*\&%\$\#@\^\~`\:\;\'\"\\\/\|]', '', text)
    text = text.lower()
    return text


cleaned_questions_train = [clean_text(q) for q in questions_train]
cleaned_answers_train = [clean_text(a) for a in answers_train]
cleaned_questions_val = [clean_text(q) for q in questions_val]
cleaned_answers_val = [clean_text(a) for a in answers_val]
cleaned_questions_test = [clean_text(q) for q in questions_test]
cleaned_answers_test = [clean_text(a) for a in answers_test]

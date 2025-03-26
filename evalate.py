import torch
import sympy

# 修改为使用你的本地模型
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

from answear import generated_answers,question

# 加载本地模型和分词器
model_path = r"D:\LLMmodel\your_local_model"  # 替换为你的本地模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)


def calculate_semantic_similarity(candidate_answer, correct_answer):
    # 对输入进行tokenize
    candidate_inputs = tokenizer(candidate_answer, return_tensors='pt', padding=True, truncation=True)
    correct_inputs = tokenizer(correct_answer, return_tensors='pt', padding=True, truncation=True)

    # 获取模型输出
    with torch.no_grad():
        candidate_output = model(**candidate_inputs).last_hidden_state.mean(dim=1)
        correct_output = model(**correct_inputs).last_hidden_state.mean(dim=1)

    # 计算余弦相似度
    similarity_score = F.cosine_similarity(candidate_output, correct_output).item()
    return similarity_score

# ... existing code ...

def check_sequence_logic(candidate_answer, sequence):
    try:
        n = sympy.symbols('n')
        formula = sympy.sympify(candidate_answer)
        for i, value in enumerate(sequence):
            if sympy.N(formula.subs(n, i + 1))!= value:
                return 0
        if len(str(formula)) < 10:  # 简单公式示例判断
            return 3
        else:
            return 5
    except Exception:
        return 0

# 假设sequence是已知数列前几项
sequence = [1, 3, 5, 7]
logic_scores = []
for candidates in generated_answers:
    scores = [check_sequence_logic(cand, sequence) for cand in candidates]
    logic_scores.append(scores)
from sklearn.metrics import accuracy_score, recall_score
import equations
from answear import generated_answers


# 假设predicted_answers是模型预测的答案列表，true_answers是真实答案列表
def evaluate_threshold(threshold, predicted_answers, true_answers):
    final_predicted = []
    for scores in predicted_answers:
        if max(scores) - min(scores) > threshold:
            final_predicted.append(scores.index(max(scores)))
        else:
            # 假设这里有反推逻辑，暂时省略具体实现，返回一个默认值
            final_predicted.append(0)
    accuracy = accuracy_score(true_answers, final_predicted)
    recall = recall_score(true_answers, final_predicted, average='macro')
    return accuracy, recall

thresholds = [5, 8, 10, 12]
results = {}
for threshold in thresholds:
    accuracy, recall = evaluate_threshold(threshold)
    results[threshold] = (accuracy, recall)

def check_linear_equation(answer, equation):
    left, right = equation.split('=')
    left = left.replace('x', answer)
    right = right.replace('x', answer)
    try:
        if eval(left) == eval(right):
            return True
        else:
            return False
    except Exception:
        return False

# 假设equations是一元一次方程列表，answers是模型生成的答案列表
validated_answers = []
for answer, equation in zip(generated_answers, equations):
    if check_linear_equation(answer, equation):
        validated_answers.append(answer)
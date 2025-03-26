from message import *
from model import tokenizer,model
import time

generated_answers = []
for question in cleaned_questions_train:
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    output = model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        num_return_sequences=3  # 简单问题生成3个答案示例
    )
    answers = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
    generated_answers.append(answers)



generated_answers_with_time = []
for question in cleaned_questions_train:
    start_time = time.time()
    input_ids = tokenizer(question, return_tensors='pt').input_ids.to(model.device)
    output = model.generate(
        input_ids,
        max_length=100,
        num_beams=5,
        num_return_sequences=3
    )
    answers = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
    end_time = time.time()
    generated_answers_with_time.append((answers, end_time - start_time))
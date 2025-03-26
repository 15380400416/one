from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from message import *
from datasets import *
import torch

# 加载模型和分词器
model_name = r"D:\LLMmodel\QwQ-32B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',torch_dtype=torch.float16
)


# 假设已经准备好微调数据集，格式为datasets.Dataset对象
# 包含'input_text'和'label'字段
train_dataset = Dataset.from_dict({
    'input_text': ['question'],
    'label': ['answer']
})

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()


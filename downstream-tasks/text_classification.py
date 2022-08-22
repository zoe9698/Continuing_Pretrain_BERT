from datasets import Dataset,load_metric
from preprocess import process
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

'''loading dataset'''
train_df = process("train")[:5000]
test_df = process("test")[:1000]
dev_df = process("dev")[:1000]
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)
dev_ds = Dataset.from_pandas(dev_df)

'''fenci'''
tokenizer = AutoTokenizer.from_pretrained("yourspath/eg./continue_bert_base/tokenizer/", use_auth_token=True)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_trainsets = train_ds.map(preprocess_function, batched=True)
tokenized_testsets = test_ds.map(preprocess_function, batched=True)
tokenized_devsets = dev_ds.map(preprocess_function, batched=True)


'''define model'''
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    num_train_epochs=5,
    weight_decay=0.01,
)

model = AutoModelForSequenceClassification.from_pretrained("yourspath/./continue_bert_base/model/", num_labels=178)
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=178)

accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_trainsets,
    eval_dataset=tokenized_devsets,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

preformance_metric = trainer.evaluate()
# print(preformance_metric)

print(tokenized_testsets)
predictions = trainer.predict(tokenized_testsets)
print(predictions)
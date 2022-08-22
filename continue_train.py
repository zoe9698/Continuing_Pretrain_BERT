import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer,
                          LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
# from dataset_defined import load_dataset
'''loading dataset'''
# train_ds, dev_ds, test_ds = load_dataset(splits=["train","dev","test"])
train_data = pd.read_csv('your/traindata/path', sep='\t') 
dev_data = pd.read_csv('your/devdata/path', sep='\t') 
data = pd.concat([train_data, dev_data]) 
data['text'] = data['text'].apply(lambda x: x.replace('\n', ''))  
# text = '\n'.join(data.text.tolist())
text_list = data.text.tolist()
text_df = pd.DataFrame(text_list)
text_df.to_csv("./text.txt", sep="\t")
# print(text_df)



'''loading model and tokenizer'''
model_name = 'bert-base-chinese'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./continue_bert_base/tokenizer')

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./text.txt",  # mention train text file here
    block_size=256)
 
valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./text.txt",  # mention valid text file here
    block_size=256)
 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./continue_bert_base_chk/argums/",  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=100,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none")
 
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)
 
trainer.train()
trainer.save_model(f'./continue_bert_base/model/')
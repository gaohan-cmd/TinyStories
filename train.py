from transformers import TrainingArguments, Trainer
import torch
from get_datasets import batch_process_func
from llama_model import load_model, kaiming_initialization
from transformers import DataCollatorForLanguageModeling


def test_batch_process_func():
    ds_train, ds_val = batch_process_func()
    print(ds_train)
    print(ds_val)


def train_model():
    ds_train, ds_val = batch_process_func()
    training_args = TrainingArguments(
        output_dir='saves',
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_steps=1000,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        lr_scheduler_type='cosine',
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        report_to=None,
        num_train_epochs=3,
        save_steps=1000,
        save_total_limit=2,
        seed=3407
    )

    model, tokenizer = load_model()
    # 现在模型是随机初始化的，为了让模型更好地收敛，选用 Kaiming 初始化
    kaiming_initialization(model)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    # 开始训练
    trainer.train()
    print('训练结束')
    # 保存模型
    # trainer.save_model('saves')


if __name__ == '__main__':
    # test_batch_process_func()
    # batch_process_func()
    train_model()
    # batch_process_func()

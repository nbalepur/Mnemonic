import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import datasets
from huggingface_hub.hf_api import HfFolder
from trl import SFTTrainer
import config
from peft import LoraConfig
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from data_loader import load_sft_data
import wandb

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['term'])):
        text = f"### Term: {example['term'][i].title()}\n### Mnemonic: {example['mnemonic'][i]}"
        output_texts.append(text)
    return {'prompt': output_texts}

def main():

    if config.params['use_wandb']:
        run_name = f'SFT_{config.params["model_nickname"]}'
        os.environ["WANDB_PROJECT"] = config.params['wandb_name']
        os.environ["WANDB_LOG_MODEL"] = run_name

    hf_token = config.params['hf_read_token']
    model_name = config.params['adj_token_model_name']
    cache_dir = config.params['cache_dir']
    adapter_name = config.params['sft_adapter_name']
    tokenizer_name = config.params['sft_tokenizer_name']
    HfFolder.save_token(hf_token)

    ds_sft_train, ds_sft_eval = load_sft_data()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        device_map="auto",
        cache_dir=cache_dir
    )

    ds_sft_train, ds_sft_eval = ds_sft_train.map(formatting_prompts_func, batched=True), ds_sft_eval.map(formatting_prompts_func, batched=True)

    if config.params['use_wandb']:
        training_args = TrainingArguments(
            num_train_epochs=5, 
            output_dir=config.params['sft_output_dir'],
            overwrite_output_dir=True,
            logging_strategy='epoch',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to='wandb',
            run_name=run_name
        )
    else:
        training_args = TrainingArguments(
            num_train_epochs=5, 
            output_dir=config.params['sft_output_dir'],
            overwrite_output_dir=True,
            logging_strategy='epoch',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            run_name=run_name
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds_sft_train,
        eval_dataset=ds_sft_eval,
        dataset_text_field="prompt",
        max_seq_length=128,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(adapter_name)
    tokenizer.save_pretrained(tokenizer_name)

if __name__ == '__main__':
    main()
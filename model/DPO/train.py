import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import config
from data_loader import load_dpo_data
from trl import DPOTrainer
import wandb

def prompt_convert(ex):
    new_prompts = []
    for i, p in enumerate(ex['prompt']):
        new_prompt = f"### Term: {p.title()}\n### Mnemonic:"
        new_prompts.append(new_prompt)
    ex['prompt'] = new_prompts
    return ex

def main():

    if config.params['use_wandb']:
        run_name = f'DPO_{config.params["model_nickname"]}'
        os.environ["WANDB_PROJECT"] = config.params['wandb_name']
        os.environ["WANDB_LOG_MODEL"] = run_name

    sft_final_model_name = config.params['sft_final_output_dir']
    adapter_name = config.params['sft_adapter_name']
    tokenizer_name = config.params['sft_tokenizer_name']
    cache_dir = config.params['cache_dir']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)

    device_map = config.device_map
    model = AutoModelForCausalLM.from_pretrained(
        sft_final_model_name,
        load_in_8bit=config.params['load_in_8bit'],
        load_in_4bit=config.params['load_in_4bit'],
        cache_dir=cache_dir,
        device_map=device_map,
    )
    model_ref = None

    ds_dpo_train, ds_dpo_eval = load_dpo_data()
    ds_dpo_train_ = ds_dpo_train.map(prompt_convert, batched=True)
    ds_dpo_eval_ = ds_dpo_eval.map(prompt_convert, batched=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if config.params['use_wandb']:
        training_args = TrainingArguments(
            num_train_epochs=5, 
            overwrite_output_dir=True,
            output_dir=config.params['dpo_output_dir'],
            metric_for_best_model='rewards/accuracies',
            logging_strategy='epoch',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            per_device_train_batch_size=1,
            learning_rate=1e-06,
            load_best_model_at_end=True,
            report_to='wandb',
            run_name=run_name
        )
    else:
        training_args = TrainingArguments(
            num_train_epochs=5, 
            overwrite_output_dir=True,
            output_dir=config.params['dpo_output_dir'],
            metric_for_best_model='rewards/accuracies',
            logging_strategy='epoch'
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=2,
            per_device_train_batch_size=1,
            learning_rate=1e-06,
            load_best_model_at_end=True,
            report_to='wandb',
            run_name=run_name
        )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        max_length=64,
        max_prompt_length=16,
        beta=0.1,
        args=training_args,
        train_dataset=ds_dpo_train_,
        eval_dataset=ds_dpo_eval_,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    dpo_trainer.train()
    dpo_adapter_name = config.params['dpo_adapter_name']
    dpo_trainer.save_model(dpo_adapter_name)

if __name__ == '__main__':
    main()
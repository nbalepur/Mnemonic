import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from peft import PeftConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import time

def main():

    print(config.params)

    start_time = time.time()

    cache_dir = config.params['cache_dir']

    sft_model_name = config.params['sft_final_output_dir']
    dpo_adapter_name = config.params['dpo_adapter_name']
    tokenizer_name = config.params['sft_tokenizer_name']
    final_model_name = config.params['dpo_final_output_dir']

    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_name,
                                                cache_dir=cache_dir,
                                                device_map="auto")
    print(f'loaded base model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    dpo_model = PeftModel.from_pretrained(sft_model, dpo_adapter_name)
    print(f'loaded PEFT model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    dpo_model = dpo_model.merge_and_unload()
    print(f'merged model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    dpo_model.save_pretrained(final_model_name)
    print(f'done saving in {time.time() - start_time} seconds', flush=True)

if __name__ == '__main__':
    main()
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
import time

def main():

    start_time = time.time()

    final_model_name = config.params['sft_final_output_dir']
    adapter_name = config.params['sft_adapter_name']
    tokenizer_name = config.params['sft_tokenizer_name']
    model_name = config.params['adj_token_model_name']
    cache_dir = config.params['cache_dir']

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                cache_dir=cache_dir,
                                                device_map="cpu")
    print(f'loaded base model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    model = PeftModel.from_pretrained(model, adapter_name, device_map='cpu')
    print(f'loaded PEFT model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    model = model.merge_and_unload()
    print(f'merged model in {time.time() - start_time} seconds', flush=True)

    start_time = time.time()
    model.save_pretrained(final_model_name)
    print(f'done saving in {time.time() - start_time} seconds', flush=True)

if __name__ == '__main__':
    main()
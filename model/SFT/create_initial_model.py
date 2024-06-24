import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

def main():

    model_name = config.params['base_model_name']
    final_model_name = config.params['adj_token_model_name']
    cache_dir = config.params['cache_dir']

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        load_in_4bit=False,
        device_map="cpu",
        cache_dir=cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model.resize_token_embeddings(len(tokenizer))

    token = config.params['hf_write_token']
    model.save_pretrained(final_model_name)
    tokenizer.save_pretrained(final_model_name)

if __name__ == '__main__':
    main()
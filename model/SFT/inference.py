import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import datasets
import tqdm
from huggingface_hub.hf_api import HfFolder
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import config
from data_loader import load_test_data
import pickle
from peft import PeftModel
import torch

def prompt_convert(ex):
    new_prompts = []
    for i, p in enumerate(ex['term']):
        new_prompt = f"### Term: {p.title()}\n### Mnemonic: {p.title()} sounds like"
        new_prompts.append(new_prompt)
    ex['prompt'] = new_prompts
    return ex

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens = [], prompt_len = 0):
        super().__init__()
        self.prompt_len = prompt_len
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        sublist = self.stop_tokens
        input_ids = input_ids[0].tolist()
        seq_in_gen = sublist in [input_ids[i:len(sublist)+i] for i in range(self.prompt_len, len(input_ids))]
        return seq_in_gen

def main():

    exit(0)

    cache_dir = config.params['cache_dir']

    sft_model_name = config.params['sft_final_output_dir']
    tokenizer_name = config.params['sft_tokenizer_name']

    ds_test = load_test_data().map(prompt_convert, batched=True)
    inf_prompts = ds_test['prompt']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_name,
                                                    load_in_8bit=config.params['load_in_8bit'],
                                                    load_in_4bit=config.params['load_in_4bit'],
                                                    cache_dir=cache_dir,
                                                    device_map="auto")

    path = '/'.join(config.params['sft_results_dir'].split('/')[:-1])
    if not os.path.exists(path):
        os.makedirs(path)

    f = config.params['sft_results_dir']
    if os.path.isfile(f):
        with open(f, 'rb') as handle:
            outputs = pickle.load(handle) 
    else:
        outputs = []

    stop_token = '\n'
    for idx in tqdm.tqdm(range(len(outputs), len(inf_prompts))):

        prompt = inf_prompts[idx]
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer(stop_token).input_ids[2:], prompt_len=input_ids.shape[1])])
        input_ids = input_ids.to('cuda')
        out = sft_model.generate(input_ids, max_new_tokens=128, do_sample=False, stopping_criteria=stopping_criteria).to('cpu').detach()
        out = out[:, input_ids.shape[1]:]
        out = tokenizer.batch_decode(out)
        outputs.append(out)

    with open(config.params['sft_results_dir'], 'wb') as handle:
        pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
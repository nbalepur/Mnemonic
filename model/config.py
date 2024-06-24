params = {
    'model_nickname': 'llama_70b', # the nickname of the model (how you want it to be saved)
    'base_model_name': 'meta-llama/Llama-2-70b-hf', # points to the huggingface model as the base model
    
    'use_wandb': True, # should we log the training to wandb?
    'wandb_key': '...', # if `use_wandb` is True, put your API key here
    'wandb_name': 'Mnemonic', # if `use_wandb` is True, give a name for the project
    'open_ai_key': '...', # OpenAI key for DSPy evaluation

    'load_in_8bit': True, # load the model in 8bit?
    'load_in_4bit': False, # load the model in 4bit?

    'cache_dir': '.../cache', # cache directory to store models, datasets, etc.
    'model_save_dir': '.../models', # directory for where the models should be saved
    'results_save_dir': '.../results', # directory for where the results should be saved
}

# specify the device_map
device_map = 'auto'

"""
NOTE: For larger LLMs like LLaMA-70b, you need a custom device map if you are using multiple GPUs.
Specifically, the device_map should map 'model.norm' and 'lm_head' to 0, or else you will get an error saying that 'cuda expected tensors to be on the same device...'
The reason this happens is because in our inference script, putting the input tokens to CUDA means that they will be on the 0th GPU, so 'model.norm' and 'lm_head' must also be, as they are the start of the LLM
We have an example for LLaMA-70B using 8 GPUs below:
"""

# device_map = {'model.embed_tokens': 1,
#  'model.layers.0': 1,
#  'model.layers.1': 1,
#  'model.layers.2': 1,
#  'model.layers.3': 1,
#  'model.layers.4': 1,
#  'model.layers.5': 1,
#  'model.layers.6': 1,
#  'model.layers.7': 1,
#  'model.layers.8': 1,
#  'model.layers.9': 1,
#  'model.layers.10': 1,
#  'model.layers.11': 1,
#  'model.layers.12': 1,
#  'model.layers.13': 1,
#  'model.layers.14': 1,
#  'model.layers.15': 1,
#  'model.layers.16': 1,
#  'model.layers.17': 1,
#  'model.layers.18': 1,
#  'model.layers.19': 1,
#  'model.layers.20': 2,
#  'model.layers.21': 2,
#  'model.layers.22': 2,
#  'model.layers.23': 2,
#  'model.layers.24': 2,
#  'model.layers.25': 2,
#  'model.layers.26': 2,
#  'model.layers.27': 2,
#  'model.layers.28': 2,
#  'model.layers.29': 2,
#  'model.layers.30': 2,
#  'model.layers.31': 3,
#  'model.layers.32': 3,
#  'model.layers.33': 3,
#  'model.layers.34': 3,
#  'model.layers.35': 3,
#  'model.layers.36': 3,
#  'model.layers.37': 3,
#  'model.layers.38': 3,
#  'model.layers.39': 3,
#  'model.layers.40': 3,
#  'model.layers.41': 3,
#  'model.layers.42': 4,
#  'model.layers.43': 4,
#  'model.layers.44': 4,
#  'model.layers.45': 4,
#  'model.layers.46': 4,
#  'model.layers.47': 4,
#  'model.layers.48': 4,
#  'model.layers.49': 4,
#  'model.layers.50': 4,
#  'model.layers.51': 4,
#  'model.layers.52': 4,
#  'model.layers.53': 5,
#  'model.layers.54': 5,
#  'model.layers.55': 5,
#  'model.layers.56': 5,
#  'model.layers.57': 5,
#  'model.layers.58': 5,
#  'model.layers.59': 5,
#  'model.layers.60': 5,
#  'model.layers.61': 5,
#  'model.layers.62': 5,
#  'model.layers.63': 5,
#  'model.layers.64': 6,
#  'model.layers.65': 6,
#  'model.layers.66': 6,
#  'model.layers.67': 6,
#  'model.layers.68': 6,
#  'model.layers.69': 6,
#  'model.layers.70': 6,
#  'model.layers.71': 6,
#  'model.layers.72': 6,
#  'model.layers.73': 6,
#  'model.layers.74': 6,
#  'model.layers.75': 0,
#  'model.layers.76': 0,
#  'model.layers.77': 0,
#  'model.layers.78': 0,
#  'model.layers.79': 0,
#  'model.norm': 0,
#  'lm_head': 0}

# ******************* these parameters are all specified automatically based on what you put above! *******************
params['adj_token_model_name'] = f'{params["model_save_dir"]}/{params["model_nickname"]}-adj'
params['sft_output_dir'] = f'{params["model_save_dir"]}/sft_{params["model_nickname"]}'
params['sft_final_output_dir'] = f'{params["model_save_dir"]}/sft_{params["model_nickname"]}_final'
params['dpo_output_dir'] = f'{params["model_save_dir"]}/dpo_{params["model_nickname"]}'
params['dpo_final_output_dir'] = f'{params["model_save_dir"]}/dpo_{params["model_nickname"]}_final'
params['sft_adapter_name'] = f'{params["model_save_dir"]}/sft_{params["model_nickname"]}_adapter'
params['sft_tokenizer_name'] =  f'{params["model_save_dir"]}/sft_{params["model_nickname"]}_tokenizer'
params['dpo_adapter_name'] = f'{params["model_save_dir"]}/dpo_{params["model_nickname"]}_adapter'
params['sft_results_dir'] = f'{params["results_save_dir"]}/{params["model_nickname"]}/sft.pkl'
params['dpo_results_dir'] = f'{params["results_save_dir"]}/{params["model_nickname"]}/dpo.pkl'
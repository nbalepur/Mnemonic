import datasets
import config

def load_sft_data():
    ds = datasets.load_dataset('nbalepur/Mnemonic_SFT', cache_dir=config.params['cache_dir'])
    return ds['train'], ds['test']

def load_dpo_data():
    ds = datasets.load_dataset('nbalepur/Mnemonic_Chosen_Rejected', cache_dir=config.params['cache_dir'])
    return ds['train'], ds['val']

def load_test_data():
    ds = datasets.load_dataset('nbalepur/Mnemonic_Test', cache_dir=config.params['cache_dir'])
    return ds['train']
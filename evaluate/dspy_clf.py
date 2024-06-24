import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import pickle
import pandas as pd
import datasets
import numpy as np
import dspy
import tqdm
import re
import nltk
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate import Evaluate
from model import config

# ********************* DSPy Signatures *********************

class MnemonicClassification(dspy.Signature):
    """Given a vocabulary term, a sentence using the term, and two candidate mnemonics (Mnemonic A and Mnemonic B), classify whether Mnemonic A or Mnemonic B is a better mnemonic device. Output just the letter of the better mnemonic ("A" or "B")"""
    
    term = dspy.InputField(desc="Vocabulary term", prefix="Term:")
    sentence = dspy.InputField(desc="Sentence showing an example of how the vocabulary word is used", prefix="Sentence:")
    mnemonic_a = dspy.InputField(desc="Mnemonic A", prefix="Mnemonic A:")
    mnemonic_b = dspy.InputField(desc="Mnemonic B", prefix="Mnemonic B:")
    answer = dspy.OutputField(desc="Answer with the letter of the mnemonic that is a better mnemonic device (A or B)", prefix="Answer:")

class MnemonicClassificationFewShot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(MnemonicClassification)

    def forward(self, term, sentence, mnemonic_a, mnemonic_b):
        return self.generate_answer(term=term, sentence=sentence, mnemonic_a=mnemonic_a, mnemonic_b=mnemonic_b)

# ********************* Text Parsing Helper Functions *********************

"""
Function to parse and clean the text in mnemonics
We remove non-alphanumeric characters and keep the first three sentences of the mnemonic (in case the model generates very long text)
"""
def parse_text(text, cutoff=3):
    pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'
        u'\U0001F300-\U0001F5FF'
        u'\U0001F680-\U0001F6FF'
        u'\U0001F700-\U0001F77F'
        u'\U0001F780-\U0001F7FF'
        u'\U0001F800-\U0001F8FF'
        u'\U0001F900-\U0001F9FF'
        u'\U0001FA00-\U0001FA6F'
        u'\U0001FA70-\U0001FAFF'
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251' 
        u'\U0001F260-\U0001F265'
        '\uFFFD'
        ']+', 
        flags=re.UNICODE)
    txt = pattern.sub('', text)
    sentences = nltk.sent_tokenize(txt)
    if len(sentences) <= cutoff:
        return txt
    return ' '.join(sentences[:cutoff])

# functions to parse the DSPy outputs
def parse_out(out):
    if out[0] == 'A' or 'Answer: A' in out:
        return 'A'
    if out[0] == 'B' or 'Answer: B' in out:
        return 'B'
    return 'Invalid'

def parse_out_swap(out):
    if out[0] == 'A' or 'Answer: A' in out:
        return 'B'
    if out[0] == 'B' or 'Answer: B' in out:
        return 'A'
    return 'Invalid'

# ********************* Main method *********************

def main():

    with open(config.params['sft_results_dir'], 'rb') as handle:
        typeA = pickle.load(handle) 
    with open(config.params['dpo_results_dir'], 'rb') as handle:
        typeB = pickle.load(handle)

    OPENAI_API_KEY = config.params['open_ai_key']
    gpt4 = dspy.OpenAI(model='gpt-4-turbo-2024-04-09', max_tokens=20, api_key=OPENAI_API_KEY)
    dspy.configure(lm=gpt4)

    ds = datasets.load_dataset('nbalepur/Mnemonic_Test', cache_dir=config.params['cache_dir'],)
    terms = ds['train']['term']
    with open('./sentence_info.pkl', 'rb') as handle:
        ex_map = pickle.load(handle)

    typeA = [[t] if type(t) == type('') else t for t in typeA]
    typeB = [[t] if type(t) == type('') else t for t in typeB]

    typeA = [parse_text(t[0].split('\n')[0]).strip() for t in typeA]
    typeB = [parse_text(t[0].split('\n')[0]).strip() for i, t in enumerate(typeB)]

    typeA = [terms[i].title() + ' sounds like ' + t for i, t in enumerate(typeA)]
    typeB = [terms[i].title() + ' sounds like ' + t for i, t in enumerate(typeB)]

    evalset_A = []
    for i in range(len(terms)):
        ex = dspy.Example(
        term=terms[i],
        sentence=ex_map.get(terms[i].lower(), ''),
        mnemonic_a=typeA[i].lower(),
        mnemonic_b=typeB[i].lower()
        )
        evalset_A.append(ex.with_inputs("term", "sentence", "mnemonic_a", "mnemonic_b"))

    evalset_B = []
    for i in range(len(terms)):
        ex = dspy.Example(
        term=terms[i],
        sentence=ex_map.get(terms[i].lower(), ''),
        mnemonic_b=typeA[i].lower(),
        mnemonic_a=typeB[i].lower()
        )
        evalset_B.append(ex.with_inputs("term", "sentence", "mnemonic_a", "mnemonic_b"))

    mnemonic_clf = MnemonicClassificationFewShot()
    mnemonic_clf.load("./prompt.json")

    predA = []
    for x in tqdm.tqdm(evalset_A):
        pred = mnemonic_clf(**x.inputs())
        predA.append(pred.answer)

    predB = []
    for x in tqdm.tqdm(evalset_B):
        pred = mnemonic_clf(**x.inputs())
        predB.append(pred.answer)

    predA = [parse_out(x) for x in predA]
    predB = [parse_out_swap(x) for x in predB]

    # save the results, and print the summary
    final_pred = predA + predB
    wins = []
    for i in range(len(predA)):
        if predA[i] != predB[i]:
            wins.append('Tie')
        else:
            wins.append(predA[i])
    metric_summary = pd.DataFrame(wins).value_counts() / len(predA)

    with open(f'./eval.pkl', 'wb') as handle:
        pickle.dump({'predA': predA, 'predB': predB, 'metric_summary': metric_summary}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(metric_summary)

if __name__ == '__main__':
    main()
# SMART Mnemonic Generation

This repository contains the code, data, and pre-trained models for our [Arxiv paper](https://arxiv.org/abs/2406.15352): **A SMART Mnemonic Sounds like “Glue Tonic”: Mixing LLMs with Student Feedback to Make Mnemonic Learning Stick**

<h3 align="center">
<span style="color:black">🦾 <a style="color:black;" href="https://huggingface.co/collections/nbalepur/mnemonic-generation-6674c357b3882fd58790ebd4">Models</a>&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;📊 <a href="https://huggingface.co/collections/nbalepur/mnemonic-generation-6674c357b3882fd58790ebd4">Data</a>&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;📝 <a href="https://arxiv.org/abs/2406.15352">Paper</a></span>
</h3>

<br />

![Mnemonic_Figure](https://github.com/nbalepur/Mnemonic/assets/55101514/de8fd5be-2a02-4d0c-a170-5e56138f3ab8)

## Abstract

Keyword mnemonics are memorable explanations that link new terms to simpler keywords.
Prior works generate mnemonics for students, but they do not guide models toward mnemonics students prefer and aid learning.
We build SMART, a mnemonic generator trained on feedback from real students learning new terms.
To train SMART, we first fine-tune LLaMA-2 on a~curated set of user-written mnemonics.
We then use LLM alignment to enhance \model: we deploy mnemonics generated by SMART in a flashcard app to find preferences on mnemonics students favor.
We gather 2684 preferences from 45 students across two types: **expressed** (inferred from ratings) and **observed** (inferred from student learning), yielding three key findings:

1. Expressed and observed preferences disagree; what students *think* is helpful does not fully capture what is *truly* helpful
2. Bayesian models can synthesize complementary data from multiple preference types into a single effectiveness signal.
SMART is tuned via Direct Preference Optimization on this signal, which we show resolves ties and missing labels in the typical method of pairwise comparisons, augmenting data for LLM output quality gains. 
3. Mnemonic experts assess SMART as matching GPT-4, at much lower deployment costs, showing the utility of capturing diverse student feedback to align LLMs in education.

## Models and Dataset

Our datasets and models can all be downloaded [here](https://huggingface.co/collections/nbalepur/mnemonic-generation-6674c357b3882fd58790ebd4). More specifically, we provide the following datasets and pre-trained models:

### Mnemonic Datasets
- [Mnemonic Fine-tuning Data](https://huggingface.co/datasets/nbalepur/Mnemonic_SFT)
- [Mnemonic Student Preferences Data](https://huggingface.co/datasets/nbalepur/Mnemonic_Pref)
- [Mnemonic Chosen/Rejected with Bayesian Labels](https://huggingface.co/datasets/nbalepur/Mnemonic_Chosen_Rejected)
- [Mnemonic Test Set](https://huggingface.co/datasets/nbalepur/Mnemonic_Test)

### Pre-trained SMART Models (LLaMA-2 70B)
- [SMART Tokenizer](https://huggingface.co/datasets/nbalepur/LLama-2-70b-Mnemonic-Tokenizer)
- [SMART Fine-tuned Model](https://huggingface.co/nbalepur/LLama-2-70b-Mnemonic-SFT)
- [SMART DPO Model](https://huggingface.co/nbalepur/LLama-2-70b-Mnemonic-DPO/)

## Code

We provide the code for training SMART (or any LLM), learning the Bayesian effectiveness labels from diverse student feedback, and evaluating the pairwise quality of two model-generated mnemonic devices 

### Training SMART

We have released the fine-tuning and DPO datasets using our combined Bayesian preference labels, so you can reproduce our trained model with the following steps.

1. Navigate to `/model/`
2. Set the specified parameters in `config.py` (described in the file)
2. Run `python SFT/create_initial_model.py`. This creates the initial LLaMA model with an extra token for padding. Requires higher CPU memory (1024 GB for LLaMA-2 70B)
3. Run `python SFT/train.py`. This trains the initial SMART model with supervised fine-tuning and LoRA. Requires higher GPU memory (192 GB for LLaMA-2 70B)
4. Run `python SFT/merge.py`. Merges the LoRA fine-tuned model into the original model. Requires higher CPU memory (1024 GB for LLaMA-2 70B)
5. Run `python DPO/train.py`. Further tunes the initial SMART model with DPO. Requires higher GPU memory (192 GB for LLaMA-2 70B)
6. Run `python DPO/merge.py`. Merges the LoRA DPO model into the original model. Requires higher CPU memory (1024 GB for LLaMA-2 70B)

All training hyperparameters are specified as the ones used when training SMART with LLaMA-2 70B. For different LLMs, different hyperparameters may be needed. We did some preliminary testing with LLaMA-2 7B and found that our current hyperparameters did not lead to large improvements with DPO (2% difference in Win/Loss Ratio versus the 10% difference in Win/Loss Ratio seen for LLaMA-2 70B), so some search may be necessary for optimal results.

Once that is all done, you can run inference with the SFT model using `python SFT/inference.py` and the DPO model using `python DPO/inference.py`. The results will be saved in a `.pkl` file in your specified `results_dir` folder.

### Bayesian Preference Labels

You can run our Bayesian model with the command `python Bayesian/bayesian.py` (while staying in the `/model` folder). We have included the preprocessed data this model expects as input in `Bayesian/bayesian_data.pkl` (derived from student preference data), and the file `bayesian.py` details what each of these fields are. After running `bayesian.py`, we aggregate the latent effectiveness across chains and epochs from NUTS to construct a preference dataset, which will be saved under `Bayesian/chosen_rejected_data` 

### Evaluation

We also provide the code for comparing the quality of two mnemonic devices with GPT-4, which can be run by navigating to the `evaluate` folder and running `dspy_clf.py`. The code is currently set up to compare the fine-tuned versus DPO-trained LLMs for mnemonic generation, but this can be changed by altering which results to load on lines 87 to 90.

## Citation

If you found our code, datasets, or paper useful, please cite:

```bibtex
@misc{balepur2024smart,
  title = {A SMART Mnemonic Sounds like "Glue Tonic": Mixing LLMs with Student Feedback to Make Mnemonic Learning Stick},
  author = {Balepur, Nishant and Shu, Matthew and Hoyle, Alexander and Robey, Alison and Feng, Shi and Goldfarb-Tarrant, Seraphina and Boyd-Graber, Jordan},
  year = {2024},
  eprint = {2406.15352},
  archiveprefix = {arXiv},
}
```

If you have any questions or problems with the code feel free to raise an issue or email me at [nbalepur@umd.edu](mailto:nbalepur@umd.edu). Thank you!
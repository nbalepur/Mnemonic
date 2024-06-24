import numpy as np
import pymc as pm
import arviz as az
import pickle
import datasets

def main():

  # load the preprocessed data (the required format for this data is detailed below under "Define Data")
  with open('./Bayesian/bayesian_data.pkl', 'rb') as handle:
      bayesian_data = pickle.load(handle)

  # run the Bayesian model
  with pm.Model() as model:

      # *************** Fixed Values ***************
      n_likert_categories = 5
      n_instances = len(bayesian_data['mnemonic_a'])

      # *************** Define Data ***************
      direct_comparisons_curr = bayesian_data['direct_comparisons'] # list containing direct comparison votes between Mnemonic A and B. Each list element is in the form [a_votes, b_votes, tie_votes]
      ratings_A_counts_curr = bayesian_data['ratings_A_counts'] # list of 5-length cumulative likert counts on Mnemonic A. Each list element is in the form [# votes leq 1, # votes leq 2, ...]
      ratings_B_counts_curr = bayesian_data['ratings_B_counts'] # same as above for Mnemonic B
      turn_counts_A_curr = bayesian_data['turn_counts_A'] # list of turns needed to learn Mnemonic A
      turn_counts_B_curr = bayesian_data['turn_counts_B'] # list of turns needed to learn Mnemonic B

      compare_idx_curr = bayesian_data['compare_idx'] # list of indexes denoting all 472 mnemonic pairs that have comparison ratings (e.g. [0, 1, 4, 6, ...])
      ratings_A_idx_curr = bayesian_data['ratings_A_idx'] # list of indexes denoting all 472 'Mnemonic A' mnemonics that have Likert ratings
      ratings_B_idx_curr = bayesian_data['ratings_B_idx'] # same as above but for Mnemonic B
      learn_A_idx_curr = bayesian_data['learn_A_idx']  # list of indexes denoting all 472 'Mnemonic A' mnemonics that have learning outcomes (turns). Since this is modeled by a geometric distribution, the indexes can be repeated and correspond to each student (e.g. [0, 0, 0, 2, 3, 3, ...])
      learn_B_idx_curr = bayesian_data['learn_B_idx'] # same as above but for Mnemonic B

      # *************** Overall Mnemonic Effectiveness ***************
      overall_effectiveness_A = pm.Beta('effectiveness_A', alpha=1, beta=1, shape=n_instances)
      overall_effectiveness_B = pm.Beta('effectiveness_B', alpha=1, beta=1, shape=n_instances)

      # *************** Comparison Data ***************
      compare_slope = pm.Normal('compare_slope', mu=0, sigma=1)
      compare_intercept = pm.Normal('compare_intercept', mu=0, sigma=1)
      compare_a_prob = pm.math.sigmoid(compare_slope * overall_effectiveness_A[compare_idx_curr] + compare_intercept)
      compare_b_prob = pm.math.sigmoid(compare_slope * overall_effectiveness_B[compare_idx_curr] + compare_intercept)

      tie_prob = pm.Beta('tie_prob', alpha=1, beta=1)
      total_prob = (compare_a_prob + compare_b_prob + tie_prob)
      comparison_probs = pm.math.stack([compare_a_prob / total_prob,
                                        compare_b_prob / total_prob,
                                        (np.ones(compare_idx_curr.shape) * tie_prob / total_prob)]).T

      comparison_votes = pm.Multinomial('comparison_votes', n=pm.math.sum(direct_comparisons_curr, axis=1), p=comparison_probs, observed=direct_comparisons_curr)

      # *************** Likert Scale Data ***************

      likert_slope = pm.Normal('likert_slope', mu=0, sigma=1, shape=(1, n_likert_categories-1))
      likert_intercept = pm.Normal('likert_intercept', mu=0, sigma=1, shape=(1, n_likert_categories-1))

      likert_logits_a = pm.math.sigmoid(likert_slope * overall_effectiveness_A[ratings_A_idx_curr][:, None] + likert_intercept)
      likert_logits_a = pm.math.concatenate([np.zeros((ratings_A_idx_curr.shape[0], 1)), likert_logits_a], axis=1)
      likert_p_a = pm.math.softmax(likert_logits_a, axis=1)
      ratings_A = pm.Multinomial('ratings_A', n=np.sum(ratings_A_counts_curr, axis=1), p=likert_p_a, observed=ratings_A_counts_curr)

      likert_logits_b = pm.math.sigmoid(likert_slope * overall_effectiveness_B[ratings_B_idx_curr][:, None] + likert_intercept)
      likert_logits_b = pm.math.concatenate([np.zeros((ratings_B_idx_curr.shape[0], 1)), likert_logits_b], axis=1)
      likert_p_b = pm.math.softmax(likert_logits_b, axis=1)
      ratings_B = pm.Multinomial('ratings_B', n=np.sum(ratings_B_counts_curr, axis=1), p=likert_p_b, observed=ratings_B_counts_curr)

      # *************** Learning Data ***************

      learn_slope = pm.Normal('learn_slope', mu=0, sigma=1)
      learn_intercept = pm.Normal('learn_intercept', mu=0, sigma=1)

      prob_learn_a = pm.math.sigmoid(learn_slope * overall_effectiveness_A[learn_A_idx_curr] + learn_intercept)
      learn_A = pm.NegativeBinomial('learn_A', p=prob_learn_a, n=1, observed=turn_counts_A_curr)

      prob_learn_b = pm.math.sigmoid(learn_slope * overall_effectiveness_B[learn_B_idx_curr] + learn_intercept)
      learn_B = pm.NegativeBinomial('learn_B', p=prob_learn_b, n=1, observed=turn_counts_B_curr)

      # *************** Training (NUTS) ***************
      trace = pm.sample(1000, tune=1000, chains=5, random_seed=[1, 2, 3, 4, 5])

  # simple strategy to aggregate the learned effectiveness --- just average over all chains and epochs (more advanced strategies like thinning could be used here)
  is_a_better = 1 * (trace.posterior['effectiveness_A'].values.mean(axis = 1) > trace.posterior['effectiveness_B'].values.mean(axis = 1))
  is_a_better = 1 * (is_a_better.sum(axis = 0) > 2.5)

  # construct the preference dataset
  terms, mn_a, mn_b = bayesian_data['term'], bayesian_data['mnemonic_a'], bayesian_data['mnemonic_b']
  winning = []
  losing = []
  for idx, label in enumerate(is_a_better):
    if label:
      winning.append(mn_a[idx])
      losing.append(mn_b[idx])
    else:
      losing.append(mn_a[idx])
      winning.append(mn_b[idx])
  new_ds = datasets.Dataset.from_dict({'prompt': terms, 'chosen': winning, 'rejected': losing})

  # compute agreement statistics
  out = []
  for chain in range(trace.posterior['effectiveness_A'].values.shape[0]):
    effectiveness_a_samples = trace.posterior['effectiveness_A'].values
    eff_a = np.mean(effectiveness_a_samples[chain], axis = 0)

    effectiveness_b_samples = trace.posterior['effectiveness_B'].values
    eff_b = np.mean(effectiveness_b_samples[chain], axis = 0)

    out.append(1 * (eff_a > eff_b))

  agreement = []
  for i in range(5):
    for j in range(i+1, 5):
      agreement.append(np.mean(out[i] == out[j]))
  print("Average Agreement Between Chains:", np.mean(agreement))

  # save the dataset
  new_ds.save_to_disk('./Bayesian/chosen_rejected_data')

if __name__ == '__main__':
    main()
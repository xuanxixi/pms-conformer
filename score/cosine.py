import numpy as np

import torch,tqdm
import torch.nn.functional as F
import pandas as pd
def asnorm(scores: pd.DataFrame,
           cohort: dict,
           nTop: int = 300,
           std_eps: float = 1e-8):
    """
    Adaptive score normalization with top n cohort samples.

        Parameter
        ---------
            scores : DataFrame
                Scores of trials are normalized with columns {score, enroll, test}.
            cohort : dict or Tensor

            nTop : int
                The number of selected samples, usually top X scoring or most
                similar samples, where X is set to be, e.g., 200 ~ 400.
                If nTop < 0: select the nTop most disimiliar scores.
                If nTop > 0: select the nTop most similiar scores.

        Return
        ------
            scores : array
                Normalized scores of trials.

        Note
        ----
            torch.topk is faster than numpy.sort.
    """
    # N = scores.shape[0]
    N = 2000
    orig_scores = scores['score'].values
    trial_enrolls = scores['enroll'].values
    trial_tests = scores['test'].values

    task = "Cohort Statistics"

    scores_mean = {}
    scores_std = {}

    for key, value in tqdm(cohort.items(), total=len(cohort), desc=task):
        if nTop < 0:
            value, _ = torch.topk(torch.from_numpy(-value), nTop, sorted=False)
            value = -value.numpy()
        else:
            value, _ = torch.topk(torch.from_numpy(value), nTop, sorted=False)
            value = value.numpy()
        m = value.mean()
        scores_mean[key] = m
        scores_std[key] = np.sum((value - m) ** 2 / abs(nTop)) ** 0.5

    task = "Normalization Statistics"

    znorm_mean = np.empty(N)
    znorm_std = np.empty(N)
    tnorm_mean = np.empty(N)
    tnorm_std = np.empty(N)

    for i in tqdm(range(N), total=N, desc=task):
        enroll, test, orig_score = trial_enrolls[i], trial_tests[i], orig_scores[i]
        znorm_mean[i] = scores_mean[enroll]
        znorm_std[i] = max(scores_std[enroll], std_eps)
        tnorm_mean[i] = scores_mean[test]
        tnorm_std[i] = max(scores_std[test], std_eps)

    znorm_scores = (orig_scores - znorm_mean) / znorm_std

    tnorm_scores = (orig_scores - tnorm_mean) / tnorm_std

    norm_scores = 0.5 * (znorm_scores + tnorm_scores)
    return norm_scores
def cosine_score(trials, index_mapping, eval_vectors):
    labels = []
    scores = []
    for item in trials:
        print("item[1]",item[1])
        print("item[2]",item[2])
        # print("index_mapping[item[1]]",index_mapping[item[1]])
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom
        labels.append(int(item[0]))
        scores.append(score)
    return labels, scores
    # return scores


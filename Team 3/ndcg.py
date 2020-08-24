
import math
import numpy as np
"""
This function calculates ndcg for multiple queries dataset

y_true : array, shape = [n_samples]
        Ground truth (true relevance labels)
y_score : array, shape = [n_samples]
        Predicted scores
k:      int, optional 
        Only consider the highest k scores in the ranking. If None, use all outputs.
group: array, shape = [n_groups]
        each element denotes how many items are there in each group
assume all queries have equal weights
"""

def ndcg_score(y_true, y_score, group, k = None):
    avg_ndcg = 0
    index = 0 #next row to be calculated
    count = 0 #number of groups which can provide information(useful group: at least one score in that group is not equal to 0)
    for i in range(0, len(group)):
        cur_true = y_true[index: index+group[i]-1]
        cur_score = y_score[index: index+group[i]-1]
        index += group[i]

        idcg = dcg_score(cur_true, cur_true, k)
        # when ground truth is equal to 0, we abandon that group which provides no information
        if idcg == 0:
            continue

        cur_ndcg = dcg_score(cur_true, cur_score, k)/idcg
        avg_ndcg = cur_ndcg * 1/(count+1) + avg_ndcg * count/(count+1)
        count += 1

    return avg_ndcg

def dcg_score(y_true, y_score, k = None):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)












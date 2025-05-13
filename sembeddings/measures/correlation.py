# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2025 Thiago César Castilho Almeida et al.

import numpy as np

def compute_jacmax(Ti, Tj, top_K):
    """
    Computes the JaccardMax similarity between two ranked lists.

    The JaccardMax score is the maximum Jaccard similarity across the top-K prefixes
    of the two lists.

    Args:
        Ti (list[int]): First ranked list.
        Tj (list[int]): Second ranked list.
        top_K (int): Maximum prefix length to consider.

    Returns:
        float: Maximum Jaccard similarity.
    """
    set_Ti = set()
    set_Tj = set()
    
    jaccard_max = 0
    for d in range(1, top_K):
        set_Ti.add(Ti[d - 1])
        set_Tj.add(Tj[d - 1])
        
        intersect = len(set_Ti & set_Tj)
        union = len(set_Ti | set_Tj)
        
        jaccard = intersect / union
        if jaccard == 1:
            return jaccard
        if jaccard > jaccard_max:
            jaccard_max = jaccard
            
    return jaccard_max

def compute_reciprocal_knn_distance(Ti, Tj, top_K, R):
    """
    Computes the reciprocal k-NN score between two ranked lists
    using a precomputed R matrix.

    Args:
        Ti (list[int]): First ranked list.
        Tj (list[int]): Second ranked list.
        top_K (int): Number of top entries to consider.
        R (np.ndarray): Precomputed reciprocal values matrix.

    Returns:
        float: Weighted reciprocal distance score.
    """
    Ti_slice = Ti[:top_K]
    Tj_slice = Tj[:top_K]
    
    weight_matrix = np.outer(np.arange(top_K, 0, -1), np.arange(top_K, 0, -1))
    weighted_distances = R[np.ix_(Ti_slice, Tj_slice)] * weight_matrix
    
    score = np.sum(weighted_distances) / (top_K ** 4)
    return score

def compute_reciprocal_values_matrix(ranked_lists, top_K):
    """
    Precomputes a reciprocal values matrix for the given ranked lists.

    The matrix R[i][j] = 1 if j is in the top-K of list i, else 0.

    Args:
        ranked_lists (list[list[int]]): Collection of ranked lists.
        top_K (int): Number of top elements to consider.

    Returns:
        np.ndarray: Reciprocal values matrix.
    """
    dataset_size = len(ranked_lists)
    R = np.zeros((dataset_size, dataset_size))
    for i in range(dataset_size):
        R[i, ranked_lists[i][:top_K]] = 1
    return R

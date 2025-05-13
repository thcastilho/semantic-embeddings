# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2025 Thiago César Castilho Almeida et al.

from .correlation import compute_jacmax

def compute_reciprocal_density(args):
    """
    Computes the Reciprocal Density score for a given query index.

    This measures how densely connected a node is within its local neighborhood,
    based on reciprocal relationships in the top-K rankings.

    Args:
        args (tuple): Contains:
            - ranked_lists (list[list[int]]): All ranked lists.
            - index (int): The query index.
            - top_K (int): Number of top entries to consider.
            - alpha (float): (Unused, but reserved for consistency).

    Returns:
        float: Normalized reciprocal density score.
    """
    ranked_lists, index, top_K, alpha = args
    
    Ti = ranked_lists[index][:top_K]
    score = 0

    for img1 in Ti:
        Tj = ranked_lists[img1][:top_K]
        for img2 in Tj:
            for k, current_img in enumerate(Ti):
                if img2 == current_img:
                    score += 1 / (k + 1)
                    break
    
    return score / (top_K ** 2)

def compute_accumulated_jacmax(args):
    """
    Computes the Accumulated JaccardMax score for a given query.

    This metric accumulates the JaccardMax similarity between the query and its
    top-K neighbors, discounted by a decay factor alpha.

    Args:
        args (tuple): Contains:
            - ranked_lists (list[list[int]]): All ranked lists.
            - index (int): The query index.
            - top_K (int): Number of top entries to consider.
            - alpha (float): Decay factor for weighting deeper neighbors.

    Returns:
        float: Normalized accumulated JaccardMax score.
    """
    ranked_lists, index, top_K, alpha = args

    Ti = ranked_lists[index]
    score = 0

    for i in range(top_K):
        Tj = ranked_lists[int(Ti[i])]
        corr = compute_jacmax(Ti, Tj, top_K)
        score += corr * pow(alpha, i)

    return score / top_K

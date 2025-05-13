###################################################################################
#                                                                                 #
# This file is an implementation of RaDE: A Rank-based Graph Embedding Approach.  #
#                                                                                 #
# RADE is free software; you can redistribute it and/or modify it under the terms #
# of the GNU Gneral Public License as published by the Free Software Foundation;  #
# either version 2 of the License, or (at your option) any later version.         #
#                                                                                 #
# RaDE is distributed in the hope that it will be useful, but WITHOUT ANY         #
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR   #
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.      #
#                                                                                 #
# You should have received a copy of the GNU General Public License along with    #
# RaDE. If not, see <http://www.gnu.org/licenses/>.                               #
#                                                                                 #
###################################################################################

import numpy as np
import math
from tqdm import tqdm
from .utils import read_ranked_lists_file

class RaDE:
    '''
    This class implements the code of RaDE.
    DE FERNANDO, F. A. ;
    PEDRONETTE, D. C. G. ;
    DE SOUSA, G. J. ;
    VALEM, L. P. ;
    GUILHERME, I. R. .
    RaDE: A Rank-based Graph Embedding Approach.
    In: 15th International Conference on Computer Vision Theory and Applications (VISAPP),
    2020, Valleta - Malta.
    '''
    def __init__(self, rks_path=None, rks_size_L=20):
        if rks_path is None:
            raise ValueError("Ranked lists (rks_path) path not set!")

        print("\tInitializing RaDE")
        self.rks_path = rks_path
        self.top_L = rks_size_L

        self.rks = read_ranked_lists_file(self.rks_path, self.top_L)
        self.dataset_size = len(self.rks)

        self.similarity_matrix_W = None
        self.transition_matrix_A = None
        self.leaders = None
        self.embeddings = None

        print("\tRaDE initialized successfully!")

    def compute_similarity(self, i, j):
        """
        Compute similarity between item i and item j based on ranking position.
        """
        ranked_list = self.rks[i]
        if j in ranked_list:
            return 1 - math.log(ranked_list.index(j) + 1, self.top_L)
        else:
            return 0

    def compute_W_matrix(self):
        """
        Compute the similarity matrix W using ranked lists.
        """
        print("\tComputing similarity matrix W...")
        W = np.zeros((self.dataset_size, self.dataset_size))
        with tqdm(total=self.dataset_size * self.dataset_size) as pbar:
            for i in range(self.dataset_size):
                for j in range(self.dataset_size):
                    W[i, j] = self.compute_similarity(i, j)
                    pbar.update(1)
        self.similarity_matrix_W = W

    def get_candidate_list(self, A, k):
        """
        Select top-k candidates based on the diagonal of the matrix A.
        """
        diagonal = np.diagonal(A)
        indexes = list(enumerate(diagonal))
        indexes.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in indexes[:k]]

    def compute_leaders(self, num_candidates=1000, num_leaders=128, t=2):
        """
        Compute the transition matrix A and select leaders using the RaDE algorithm.
        """
        print("\tComputing transition matrix A and selecting leaders...")

        A = self.similarity_matrix_W.copy()
        for _ in range(1, t):
            A = np.dot(A, self.similarity_matrix_W)

        self.transition_matrix_A = A

        candidate_list = self.get_candidate_list(A, num_candidates)
        leaders = []
        penalties = np.ones(len(A))

        for _ in range(num_leaders):
            candidates_with_scores = [
                (A[i, i] / penalties[i], i)
                for i in candidate_list if i not in leaders
            ]
            candidates_with_scores.sort(reverse=True)

            new_leader = candidates_with_scores[0][1]
            leaders.append(new_leader)

            for i in candidate_list:
                if i != new_leader:
                    penalties[i] += A[i, new_leader]

        self.leaders = leaders

    def get_leaders(self):
        """
        Return the list of leaders.

        Raises:
            ValueError: If leaders have not been computed yet.
        """
        if self.leaders is None:
            raise ValueError(
                "Leaders have not been computed yet. "
                "Call compute_leaders(num_leaders) or pass num_leaders explicitly."
            )
        return self.leaders

    def generate_embeddings(self):
        """
        Generate embedding vectors for each item based on transition matrix and leaders.
        """
        if self.transition_matrix_A is None or self.leaders is None:
            raise ValueError("Leaders or transition matrix not computed. Run compute_leaders() first.")

        print("\tGenerating embeddings...")
        embeddings = np.zeros((self.dataset_size, len(self.leaders)))
        for i in range(self.dataset_size):
            for j, leader in enumerate(self.leaders):
                embeddings[i, j] = self.transition_matrix_A[i, leader]

        self.embeddings = embeddings

    def fit(self, num_candidates=1000, num_leaders=128, t=2):
        """
        Fit the RaDE model to compute embeddings from the ranked lists.

        Args:
            num_candidates (int): Number of candidates to consider when selecting leaders.
            num_leaders (int): Number of leaders to select.
            t (int): Number of transition steps (matrix powers) to apply.

        Returns:
            np.ndarray: Embedding matrix of shape (dataset_size, num_leaders).
        """
        self.compute_W_matrix()
        self.compute_leaders(num_candidates, num_leaders, t)
        self.generate_embeddings()

        return self.embeddings
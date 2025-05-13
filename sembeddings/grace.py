#########################################################################################
#                                                                                       #
# This file is an implementation of GRaCE: Graph and Rank-based Contextual Embeddings.  #
#                                                                                       #
# GRaCE is free software; you can redistribute it and/or modify it under the terms of   #
# of the GNU General Public License as published by the Free Software Foundation;       #
# either version 2 of the License, or (at your option) any later version.               #
#                                                                                       #
# GRaCE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;    #
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR      #
# PURPOSE. See the GNU General Public License for more details.                         #
#                                                                                       #
# You should have received a copy of the GNU General Public License along with GRaCE.   #
# If not, see <http://www.gnu.org/licenses/>.                                           #
#                                                                                       #
#########################################################################################

import numpy as np
import math
from tqdm import tqdm
from .utils import read_ranked_lists_file
from .measures.qpp import compute_reciprocal_density, compute_accumulated_jacmax
from .measures.correlation import compute_jacmax, compute_reciprocal_knn_distance, compute_reciprocal_values_matrix

class GRaCE:
    '''
    This class implements the code of GRaCE.
    ALMEIDA, T. C. C. ;
    LETÍCIO, G. R. ;
    VALEM, L. P. ;
    FREITAS, A. ;
    PEDRONETTE, D. C. G. .
    Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data
    In: 2025 International Joint Conference on Neural Networks (IJCNN),
    2025, Rome - Italy.
    '''
    def __init__(self, rks_path=None, top_K=20, correlation_measure='jacmax', estimation_measure='reciprocal_density', alpha=0.95):
        if rks_path is None:
            raise ValueError("Ranked lists (rks_path) path not set!")

        print("\tInitializing GRaCE")
        self.rks_path = rks_path
        self.top_K = top_K
        self.alpha = alpha

        self.rks = read_ranked_lists_file(self.rks_path, self.top_K)
        self.dataset_size = len(self.rks)

        self.correlation_measure = correlation_measure
        self.correlation_matrix = {}  # Dictionary to store correlation values
        self.R = None  # Required only for 'reciprocal' correlation
        self.estimation_measure = estimation_measure
        self.estimations = None
        self.leaders = None
        self.embeddings = None

        print("\tGRaCE initialized successfully!")

    def compute_estimations(self):
        """
        Compute the quality predictions (estimations) sequentially.

        Supported estimation measures:
            - 'accjacmax': Accumulated JacMax
            - 'reciprocal_density': Reciprocal Density

        Returns:
            None
        """
        print("\tComputing estimations sequentially")
        args = [(self.rks, i, self.top_K, self.alpha) for i in range(len(self.rks))]

        if self.estimation_measure == 'accjacmax':
            estimations = [compute_accumulated_jacmax(arg) for arg in tqdm(args)]
        elif self.estimation_measure == 'reciprocal_density':
            estimations = [compute_reciprocal_density(arg) for arg in tqdm(args)]
        else:
            raise ValueError(f"Invalid estimation measure: {self.estimation_measure}")

        self.estimations = estimations
    
    def get_correlation(self, i, j):
        """
        Return the correlation between i and j, using symmetric caching.

        Returns:
            float: Correlation score.
        """
        if self.correlation_measure == 'jacmax':
            key = (min(i, j), max(i, j))
        elif self.correlation_measure == 'reciprocal':
            key = (i, j)
        else:
            raise ValueError(f"Invalid correlation measure: {self.correlation_measure}")
        
        if key not in self.correlation_matrix:
            if self.correlation_measure == 'jacmax':
                value = compute_jacmax(self.rks[i], self.rks[j], self.top_K)
            elif self.correlation_measure == 'reciprocal':
                if self.R is None:
                    self.R = compute_reciprocal_values_matrix(self.rks, self.top_K)
                value = compute_reciprocal_knn_distance(self.rks[i], self.rks[j], self.top_K, self.R)
            else:
                raise ValueError(f"Invalid correlation measure: {self.correlation_measure}")
            # Cache symmetrically
            self.correlation_matrix[key] = value

        return self.correlation_matrix[key]

    def compute_leaders(self, num_leaders: int = 128):
        """
        Select leader elements based on estimation values and penalized correlation.

        Args:
            num_leaders (int): Number of leaders to select.
        """
        print("\tComputing leaders")
        penalties = [1.0] * self.dataset_size
        leaders = []

        for _ in tqdm(range(num_leaders)):
            # 1) Select the candidate with the highest estimation[i] / penalty[i]
            candidates = [
                (self.estimations[i] / penalties[i], i)
                for i in range(self.dataset_size) if i not in leaders
            ]
            selected_leader = max(candidates, key=lambda x: x[0])[1]
            leaders.append(selected_leader)

            # 2) Update penalties for all nodes
            for i in range(self.dataset_size):
                penalties[i] += self.get_correlation(selected_leader, i)

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
        Generate embedding vectors based on correlations to leaders.
        """
        print("\tGenerating embeddings")
        emb = np.zeros((self.dataset_size, len(self.leaders)))
        for i in range(self.dataset_size):
            for idx, ld in enumerate(self.leaders):
                emb[i, idx] = self.get_correlation(i, ld)
        self.embeddings = emb
    
    def fit(self, num_leaders: int = 128):
        """
        Compute estimations and select leaders.
        """
        self.compute_estimations()
        self.compute_leaders(num_leaders=num_leaders)

    def transform(self):
        """
        Generate and return the embedding matrix.
        """
        if self.estimations is None or self.leaders is None:
            raise ValueError("Call fit() before transform().")
        self.generate_embeddings()
        return self.embeddings

    def fit_transform(self, num_leaders=128):
        """
        Fit model and return embeddings.
        """
        self.fit(num_leaders=num_leaders)
        return self.transform()

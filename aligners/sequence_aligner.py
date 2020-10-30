from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SequenceAligner(ABC):
    def __init__(self, score_table):
        self._score_table = score_table
        self._score_matrix = score_table.to_numpy()

    def align_sequences(self, seq1, seq2):
        parse_to_num = pd.Series(np.arange(len(self._score_table)), self._score_table.keys())
        num_seq1 = parse_to_num[list(seq1)].values
        num_seq2 = parse_to_num[list(seq2)].values

        cost, trace = self._calculate_solution_matrices(num_seq1, num_seq2)

        pd_cost = pd.DataFrame(cost, columns=list(seq2), index=list(seq1))
        pd_trace = pd.DataFrame(trace, columns=list(seq2), index=list(seq1))

        return self._get_alignment_by_trace(pd_trace, pd_cost)

    @abstractmethod
    def _calculate_solution_matrices(self, seq1, seq2):
        pass

    @abstractmethod
    def _get_alignment_by_trace(self, pd_trace, pd_cost):
        pass



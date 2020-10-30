import numpy as np

from aligners.direction import Direction
from aligners.sequence_aligner import SequenceAligner


class GlobalSequenceAligner(SequenceAligner):

    # @jit(nopython=True, parallel=True)
    def _calculate_solution_matrices(self, seq1, seq2):
        # assuming all are converted to integers
        cost = np.zeros((len(seq1), len(seq2)), dtype=np.int32)
        trace = np.zeros((len(seq1), len(seq2)), dtype=np.int8)

        cost[0, 1:] = np.cumsum(self._score_matrix[seq2[1:], -1])
        cost[1:, 0] = np.cumsum(self._score_matrix[seq1[1:], -1])
        trace[0, 1:] = Direction.LEFT.value
        trace[1:, 0] = Direction.UP.value

        for i in range(1, len(seq1)):
            for j in range(1, len(seq2)):
                x, y = seq1[i], seq2[j]
                opts = {
                    Direction.UP.value: cost[i - 1, j] + self._score_matrix[x, -1],
                    Direction.LEFT.value: cost[i, j - 1] + self._score_matrix[-1, y],
                    Direction.DIAG.value: cost[i - 1, j - 1] + self._score_matrix[x, y]
                }

                cost[i, j] = max(opts.values())
                trace[i, j] = max(opts, key=opts.get)
        return cost, trace

    def _get_alignment_by_trace(self, pd_trace, pd_cost):
        arr1 = []
        arr2 = []

        i, j = pd_trace.shape[0] - 1, pd_trace.shape[1] - 1
        score = pd_cost.iloc[i, j]

        while i + j > 0:
            cur = pd_trace.iloc[i, j]

            if cur == Direction.LEFT.value:
                arr1.append('-')
                arr2.append(pd_trace.columns[j])
                j -= 1
            elif cur == Direction.UP.value:
                arr1.append(pd_trace.index[i])
                arr2.append('-')
                i -= 1
            else:
                arr1.append(pd_trace.index[i])
                arr2.append(pd_trace.columns[j])
                i -= 1
                j -= 1

        algn1 = ''.join(arr1)[::-1]
        algn2 = ''.join(arr2)[::-1]
        return algn1, algn2, score
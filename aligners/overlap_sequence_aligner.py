import numpy as np

from aligners.direction import Direction
from aligners.sequence_aligner import SequenceAligner


class OverlapSequenceAligner(SequenceAligner):

    # @jit(nopython=True, parallel=True)
    def _calculate_solution_matrices(self, seq1, seq2):
        # assuming all are converted to integers
        cost = np.zeros((len(seq1), len(seq2)), dtype=np.int32)
        trace = np.zeros((len(seq1), len(seq2)), dtype=np.int8)

        cost[0, 1:] = 0
        cost[1:, 0] = np.cumsum(self._score_matrix[seq2[1:], -1])
        trace[0, 1:] = Direction.LEFT.value
        trace[1:, 0] = Direction.UP.value
        trace[0, 0] = Direction.STOP.value

        for i in range(1, len(seq1)):
            for j in range(1, len(seq2)):
                x, y = seq1[i], seq2[j]

                opts = {
                    Direction.DIAG.value: cost[i - 1, j - 1] + self._score_matrix[x, y],
                    Direction.UP.value: cost[i - 1, j] + self._score_matrix[x, -1],
                    Direction.LEFT.value: cost[i, j - 1] + self._score_matrix[-1, y],
                }

                cost[i, j] = max(opts.values())
                trace[i, j] = max(opts, key=opts.get)

        return cost, trace

    def _get_alignment_by_trace(self, pd_trace, pd_cost):
        arr1 = []
        arr2 = []

        # i, j = pd_trace.shape[0] - 1, pd_trace.shape[1] - 1
        j = pd_trace.shape[1] - 1
        i = pd_cost.to_numpy()[1:, -1].argmax() + 1
        orig_i = i
        orig_j = j
        score = pd_cost.iloc[i, j]

        end_of_alignment_reached = False

        while not end_of_alignment_reached:
            cur = pd_trace.iloc[i, j]
            if cur == Direction.LEFT.value:
                arr1.append('-')
                arr2.append(pd_trace.columns[j])
                j -= 1
            elif cur == Direction.UP.value:
                arr1.append(pd_trace.index[i])
                arr2.append('-')
                i -= 1
            elif cur == Direction.DIAG.value:
                arr1.append(pd_trace.index[i])
                arr2.append(pd_trace.columns[j])
                i -= 1
                j -= 1
            else:
                end_of_alignment_reached = True

        algn1 = ''.join(arr1)[::-1]
        algn2 = ''.join(arr2)[::-1]
        algn2 += '-' * (pd_trace.shape[0] - orig_i -1)
        len_match = -pd_trace.shape[0] + orig_i + 1
        rest_of_seq1 = pd_trace.index[len(pd_trace.index) - len_match:]
        algn1 += "".join(rest_of_seq1)

        return algn1, algn2, score

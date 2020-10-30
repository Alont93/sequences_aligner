import argparse
import numpy as np
from itertools import groupby
import pandas as pd
import os
from numba import jit
import time

from aligners.global_sequence_aligner import GlobalSequenceAligner
from aligners.local_sequence_aligner import LocalSequenceAligner
from aligners.overlap_sequence_aligner import OverlapSequenceAligner


def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    In Ex1 you may assume the fasta files contain only one sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def print_aligns(algn1, algn2):
    for i in range(0, len(algn1), 50):
        print(algn1[i:i + 50])
        print(algn2[i:i + 50])
        print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('seq_a', help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('--align_type', help='Alignment type (e.g. local)', required=True)
    parser.add_argument('--score', help='Score matrix in.tsv format (default is score_matrix.tsv) ', default='score_matrix.tsv')

    command_args = parser.parse_args()

    _, seq1 = next(fastaread(command_args.seq_a))
    _, seq2 = next(fastaread(command_args.seq_b))
    score_mat = pd.read_csv(command_args.score, sep='\t', index_col=0)

    # to remove:
    # score_mat = pd.read_csv('score_matrix.tsv', sep='\t', index_col=0)
    # seq1 = 'AAAATTTT'
    # seq2 = 'TTTTAAAA'

    seq1 = '-' + seq1
    seq2 = '-' + seq2

    if command_args.align_type == 'global':
        aligner = GlobalSequenceAligner(score_mat)
    elif command_args.align_type == 'local':
        aligner = LocalSequenceAligner(score_mat)
    elif command_args.align_type == 'overlap':
        aligner = OverlapSequenceAligner(score_mat)
    else:
        return

    a1, a2, score = aligner.align_sequences(seq1, seq2)
    print_aligns(a1, a2)
    print(f'{command_args.align_type}:{score}')


if __name__ == '__main__':
    main()
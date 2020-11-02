import os
import re
import subprocess

GLOBAL_TEST_CASES = [
    ("AATTGC", "AATTGC", 30, "AATTGC", "AATTGC"),
    ("AATTG", "AATTG", 25, "AATTG", "AATTG"),
    ("AT", "A", -3, "AT", "A-"),
    ("A", "AT", -3, "A-", "AT"),
    ("CATTCAG", "GCTGCGAG", 0, "CATTC-AG", "GCTGCGAG"),
    ("GCTGCGAG", "CATTCAG", 0, "GCTGCGAG", "CATTC-AG"),
    ("ATC", "AGC", 6, "ATC", "AGC"),
    ("ATC", "TCG", -6, "ATC-", "-TCG"),
    ("ATCGATCG", "ACAC", -12, "ATCGATCG", "A-C-A-C-"),
    ("ATCGATCG", "GATCG", 1, "ATCGATCG", "---GATCG"),
]

OVERLAP_TEST_CASES = [
    ("AATTGC", "AATTGC", 30, "AATTGC", "AATTGC"),
    ("AATTG", "AATTG", 25, "AATTG", "AATTG"),
    ("AT", "A", 5, "AT", "A-"),
    ("A", "AT", 5, "A-", "AT"),
    ("CATTCAG", "GCTTCGAG", 14, "-CATTC-AG", "GC-TTCGAG"),
    ("GCTTCGAG", "CATTCAG", 14, "GC-TTCGAG", "-CATTC-AG"),
    ("ATC", "AGC", 6, "ATC", "AGC"),
    ("AGC", "ATC", 6, "AGC", "ATC"),
    ("ATC", "TCG", 10, "ATC-", "-TCG"),
    ("TCG", "ATC", 10, "-TCG", "ATC-"),
    ("TTTCCC", "AAATTT", 15, "---TTTCCC", "AAATTT---"),
    ("ATCGATCG", "ACAC", 2, "--ATCGATCG", "ACA-C-----"),
    ("ACAC", "ATCGATCG", 2, "ACA-C-----", "--ATCGATCG"),
    ("ATCGATCG", "GATCG", 25, "ATCGATCG", "---GATCG"),
    ("GATCG", "ATCGATCG", 25, "---GATCG", "ATCGATCG"),
]

LOCAL_TEST_CASES = [
    ("AATTGC", "AATTGC", 30, "AATTGC", "AATTGC"),
    ("AATTG", "AATTG", 25, "AATTG", "AATTG"),
    ("AT", "A", 5, "A", "A"),
    ("A", "AT", 5, "A", "A"),
    ("CATTCAG", "GCTTCGAG", 17, "TTC-AG", "TTCGAG"),
    ("GCTTCGAG", "CATTCAG", 17, "TTCGAG", "TTC-AG"),
    ("ATC", "AGC", 6, "ATC", "AGC"),
    ("AGC", "ATC", 6, "AGC", "ATC"),
    ("ATC", "TCG", 10, "TC", "TC"),
    ("TCG", "ATC", 10, "TC", "TC"),
    ("ATCGATCG", "GATCG", 25, "GATCG", "GATCG"),
    ("GATCG", "ATCGATCG", 25, "GATCG", "GATCG"),
    ("GAGATCGAGATT", "TCTCGATAGCGC", 21, "TCGAGA", "TCGATA"),
    ("TATGTCCGATAGCGC", "GAGATGTGATATT", 24, "ATGTCCGATA", "ATGT--GATA"),
]

OUTPUT_TEST_CASE = (
    "CCCCCCCCCACCCCCCCCCCGGGGGGGGGGTTTTTTTTTTCCCCCCCCCCGGGGGGGGGGTTTTTTTTTT",
    "A", 5,
    "CCCCCCCCCACCCCCCCCCCGGGGGGGGGGTTTTTTTTTTCCCCCCCCCCGGGGGGGGGGTTTTTTTTTT",
    "---------A------------------------------------------------------------"
)

FILE_A = ".a.fasta"
FILE_B = ".b.fasta"
SEQ_ALIGN = "./seq_align.py"


def write_files(seq_a, seq_b):
    with open(FILE_A, "w") as fasta_file:
        fasta_file.write(">this is test file {}\n".format(FILE_A))
        fasta_file.write(seq_a)

    with open(FILE_B, "w") as fasta_file:
        fasta_file.write(">this is test file {}\n".format(FILE_B))
        fasta_file.write(seq_b)


def setup():
    if os.path.exists(FILE_A):
        os.remove(FILE_A)
    if os.path.exists(FILE_B):
        os.remove(FILE_B)


def parse_res(res_output):
    seq_re = re.compile(r'^[AGCT\-]{1,50}$')
    result_re = re.compile(r'^(?P<type>(global|local|overlap))[ \t]*:[ \t]*(?P<score>-?\d+)$')
    is_a = True
    result_reached = False
    seq_a = ""
    seq_b = ""
    align_type = None
    score = None

    for idx, line in enumerate(res_output.split(os.linesep)):
        if line == "":
            continue

        assert not result_reached, os.linesep.join([
            "Found unexpected lines after result line", "Result output is:", res_output])

        sequence = seq_re.match(line)
        result = result_re.match(line)

        assert sequence or result, os.linesep.join([
            "Output is not in expected format (line - {})".format(idx + 1), "Result output is:", res_output])

        if result:
            result_reached = True
            align_type = result.groupdict()["type"]
            score = int(result.groupdict()["score"])
            continue

        if is_a:
            seq_a += sequence.string
        else:
            seq_b += sequence.string
        is_a = not is_a

    return seq_a, seq_b, align_type, score


def check_output(res_output, align_type, expected_score, expected_a, expected_b, ignore_score):
    seq_a, seq_b, found_type, score = parse_res(res_output)

    assert seq_a == expected_a, os.linesep.join(["Mismatch in seq a", "Result output is:", res_output])
    assert seq_b == expected_b, os.linesep.join(["Mismatch in seq b", "Result output is:", res_output])
    assert found_type == align_type, os.linesep.join(["Mismatch in align type", "Result output is:", res_output])
    if not ignore_score:
        assert expected_score == score, os.linesep.join(["Mismatch in score", "Result output is:", res_output])


def run_test(test_case, align_type, ignore_score=False):
    cmd = ["python", SEQ_ALIGN, FILE_A, FILE_B, "--align_type", align_type, "--score", "score_matrix.tsv"]

    write_files(test_case[0], test_case[1])

    res = subprocess.run(cmd, capture_output=True)

    assert "err" not in res.stderr.decode("utf-8").lower()

    check_output(res.stdout.decode("utf-8"), align_type, test_case[2], test_case[3], test_case[4],
                 ignore_score=ignore_score)


def test_output():
    for align_type in ("global", "overlap"):
        run_test(OUTPUT_TEST_CASE, align_type, ignore_score=True)


def test_global():
    for test_case in GLOBAL_TEST_CASES:
        run_test(test_case, "global")


def test_overlap():
    for test_case in OVERLAP_TEST_CASES:
        run_test(test_case, "overlap")


def test_local():
    for test_case in LOCAL_TEST_CASES:
        run_test(test_case, "local")


if __name__ == '__main__':
    test_overlap()
    print("done!")
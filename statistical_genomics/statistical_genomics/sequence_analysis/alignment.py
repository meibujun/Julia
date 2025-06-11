from Bio import pairwise2
from Bio.pairwise2 import format_alignment # For potential direct use, though custom formatting is also requested.

def perform_global_alignment(
    seq1: str,
    seq2: str,
    match_score: float = 1.0,
    mismatch_penalty: float = -1.0,
    open_gap_penalty: float = -2.0,
    extend_gap_penalty: float = -0.5
) -> list[str]:
    """
    Performs global alignment on two sequences using the Needleman-Wunsch algorithm
    as implemented in Biopython's pairwise2 module.

    Args:
        seq1: The first sequence (string).
        seq2: The second sequence (string).
        match_score: Score for a match.
        mismatch_penalty: Penalty for a mismatch (should be negative or zero).
        open_gap_penalty: Penalty for opening a gap (should be negative or zero).
        extend_gap_penalty: Penalty for extending a gap (should be negative or zero).

    Returns:
        A list of strings, where each string is a formatted representation of an
        optimal global alignment. If no alignment is found (though pairwise2 usually
        finds one), it returns an empty list.
        The format includes the alignment score and the aligned sequences.
    """
    # Biopython's pairwise2.align.globalms returns alignments with detailed scores:
    # (seqA_aligned, seqB_aligned, score, begin, end)
    # For global alignment, begin is usually 0 and end is length of alignment.
    alignments = pairwise2.align.globalms(
        seq1,
        seq2,
        match_score,
        mismatch_penalty,
        open_gap_penalty,
        extend_gap_penalty
    )

    formatted_alignments = []
    if alignments:
        # As per prompt, could return just the first/best.
        # For now, let's format all top-scoring ones found.
        # Often, globalms returns multiple alignments if they have the same top score.
        for alignment in alignments:
            algn1, algn2, score, begin, end = alignment
            # Using custom format as requested, format_alignment is also an option
            # formatted_str = format_alignment(*alignment) # Biopython's default formatter
            formatted_str = f"Alignment Score: {score}\n{algn1}\n{algn2}"
            formatted_alignments.append(formatted_str)
    return formatted_alignments

def perform_local_alignment(
    seq1: str,
    seq2: str,
    match_score: float = 1.0,
    mismatch_penalty: float = -1.0,
    open_gap_penalty: float = -2.0,
    extend_gap_penalty: float = -0.5
) -> list[str]:
    """
    Performs local alignment on two sequences using the Smith-Waterman algorithm
    as implemented in Biopython's pairwise2 module.

    Args:
        seq1: The first sequence (string).
        seq2: The second sequence (string).
        match_score: Score for a match.
        mismatch_penalty: Penalty for a mismatch (should be negative or zero).
        open_gap_penalty: Penalty for opening a gap (should be negative or zero).
        extend_gap_penalty: Penalty for extending a gap (should be negative or zero).

    Returns:
        A list of strings, where each string is a formatted representation of an
        optimal local alignment. If no alignment is found, it returns an empty list.
        The format includes the alignment score and the aligned sequences.
    """
    # Biopython's pairwise2.align.localms returns alignments with detailed scores:
    # (seqA_aligned, seqB_aligned, score, begin, end)
    # For local alignment, begin and end indicate the part of the sequences involved.
    alignments = pairwise2.align.localms(
        seq1,
        seq2,
        match_score,
        mismatch_penalty,
        open_gap_penalty,
        extend_gap_penalty
    )

    formatted_alignments = []
    if alignments:
        for alignment in alignments:
            algn1, algn2, score, begin, end = alignment
            # formatted_str = format_alignment(*alignment) # Biopython's default formatter
            formatted_str = f"Alignment Score: {score}\n{algn1}\n{algn2}"
            formatted_alignments.append(formatted_str)
    return formatted_alignments

# Example Usage:
# if __name__ == '__main__':
#     seq1_global = "GATTACA"
#     seq2_global = "GCATGCU"
#     print("Global Alignments:")
#     global_aligns = perform_global_alignment(seq1_global, seq2_global, 1, -1, -2, -0.5)
#     for al in global_aligns:
#         print(al)
#         print("-" * 20)

#     seq1_local = "AGCTAGCTAGCT"
#     seq2_local = "TTAGCTAGTT" # Common part "TAGCTA" or "AGCTAG"
#     print("\nLocal Alignments:")
#     local_aligns = perform_local_alignment(seq1_local, seq2_local, 2, -1, -1, -0.5) # Match=2 to emphasize match
#     for al in local_aligns:
#         print(al)
#         print("-" * 20)

#     s1 = "AGT"
#     s2 = "AGT"
#     print("\nGlobal identical:")
#     ga_ident = perform_global_alignment(s1,s2)
#     for al in ga_ident: print(al)

#     print("\nLocal identical:")
#     la_ident = perform_local_alignment(s1,s2)
#     for al in la_ident: print(al)

#     print("\nLocal substring:")
#     s_super = "XXXYYYZZZ"
#     s_sub = "YYY"
#     la_sub = perform_local_alignment(s_super, s_sub, match_score=5)
#     for al in la_sub: print(al)

def read_fasta(filepath: str) -> dict[str, str]:
    """
    Reads a FASTA formatted file and returns a dictionary of sequences.

    Args:
        filepath: The path to the FASTA file.

    Returns:
        A dictionary where keys are sequence headers (without the '>' symbol)
        and values are the corresponding sequences.
    """
    sequences = {}
    current_header = None
    current_sequence = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences[current_header] = ''.join(current_sequence)
                current_header = line[1:]
                current_sequence = []
            elif current_header:  # Only append if we are inside a sequence block
                current_sequence.append(line)

        # Add the last sequence after the loop
        if current_header:
            sequences[current_header] = ''.join(current_sequence)

    return sequences

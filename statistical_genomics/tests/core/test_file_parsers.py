import unittest
import tempfile
import os
from statistical_genomics.statistical_genomics.core.file_parsers import read_fasta

class TestFastaParser(unittest.TestCase):

    def test_read_fasta_simple(self):
        # Create a temporary FASTA file
        fasta_content = """>seq1
ATGCGTAGCATCGATCG
CGATCGATCGATCGATC
>seq2
GATTACAGATTACAGAT
TACA
"""
        # Use NamedTemporaryFile to ensure it's handled correctly across OS
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".fasta") as tmp_fasta:
            tmp_fasta_path = tmp_fasta.name
            tmp_fasta.write(fasta_content)

        expected_sequences = {
            "seq1": "ATGCGTAGCATCGATCG"+"CGATCGATCGATCGATC",
            "seq2": "GATTACAGATTACAGAT"+"TACA"
        }

        # Parse the temporary FASTA file
        parsed_sequences = read_fasta(tmp_fasta_path)

        # Assert that the parsed sequences are correct
        self.assertEqual(parsed_sequences, expected_sequences)

        # Clean up the temporary file
        os.remove(tmp_fasta_path)

if __name__ == '__main__':
    unittest.main()

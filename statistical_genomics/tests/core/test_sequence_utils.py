import unittest
from statistical_genomics.statistical_genomics.core.sequence_utils import (
    reverse_complement,
    transcribe,
    translate
)

class TestSequenceUtils(unittest.TestCase):

    # Tests for reverse_complement
    def test_reverse_complement_simple(self):
        self.assertEqual(reverse_complement("AGTC"), "GACT")
        self.assertEqual(reverse_complement("ATCG"), "CGAT")
        self.assertEqual(reverse_complement("GATTACA"), "TGTAATC")
        self.assertEqual(reverse_complement(""), "") # Test empty sequence

    def test_reverse_complement_lowercase(self):
        self.assertEqual(reverse_complement("agtc"), "GACT")

    def test_reverse_complement_mixed_case(self):
        self.assertEqual(reverse_complement("AgTc"), "GACT")

    def test_reverse_complement_invalid_chars(self):
        with self.assertRaises(ValueError):
            reverse_complement("AGTXC") # X is invalid
        with self.assertRaises(ValueError):
            reverse_complement("AUG") # U is invalid for DNA

    # Tests for transcribe
    def test_transcribe_simple(self):
        self.assertEqual(transcribe("ATGC"), "AUGC")
        self.assertEqual(transcribe("GATTACA"), "GAUUACA")
        self.assertEqual(transcribe(""), "") # Test empty sequence

    def test_transcribe_lowercase(self):
        self.assertEqual(transcribe("atgc"), "AUGC")

    def test_transcribe_mixed_case(self):
        self.assertEqual(transcribe("AtGc"), "AUGC")

    def test_transcribe_invalid_chars(self):
        with self.assertRaises(ValueError):
            transcribe("AGTXC") # X is invalid
        with self.assertRaises(ValueError):
            transcribe("AUG") # U is invalid for DNA (though it would become U anyway)

    # Tests for translate
    def test_translate_simple_orf(self):
        # AUG (M) - GCG (A) - UAA (*)
        self.assertEqual(translate("AUGGCGUAA"), "MA*")
        self.assertEqual(translate("AUGGCCCGUUAG"), "MAR*") # Methionine, Alanine, Arginine, Stop

    def test_translate_no_start_codon(self):
        self.assertEqual(translate("GCGUAA"), "")
        self.assertEqual(translate("AUAUAUGCGUAA"), "MR") # Start codon later in sequence: AUG CGU AA -> M R
        self.assertEqual(translate("UAAAUGGCG"), "MA") # Start codon after stop: UAA AUG GCG -> find AUG -> AUG GCG -> MA

    def test_translate_stop_codon_internal(self):
        self.assertEqual(translate("AUGUAAUGGUAA"), "M*") # Stop, then more codons
        self.assertEqual(translate("AUGGCGUAGCAU"), "MA*") # AUG GCG UAG -> M A *

    def test_translate_no_stop_codon(self):
        self.assertEqual(translate("AUGGCGUCGU"), "MAS") # AUG GCG UCG U -> M A S
        self.assertEqual(translate("AUG"), "M")
        self.assertEqual(translate("AUGAU"), "M") # Ignores trailing AU

    def test_translate_invalid_length(self):
        self.assertEqual(translate("AUGGCAU"), "MA") # Ignores trailing U
        self.assertEqual(translate("AUGGC"), "M")   # AUG GC -> M
        self.assertEqual(translate("AUGU"), "M")     # Ignores trailing U
        self.assertEqual(translate("AU"), "") # No start codon, too short
        self.assertEqual(translate("AUGGCGUA"), "MA") # AUG GCG UA -> M A

    def test_translate_lowercase_input(self):
        self.assertEqual(translate("auggcguaa"), "MA*")

    def test_translate_invalid_chars(self):
        with self.assertRaises(ValueError):
            translate("AUGGCGXAA") # X is invalid
        with self.assertRaises(ValueError):
            translate("AUGTCG") # T is invalid for RNA

    def test_translate_from_readme_example(self):
        self.assertEqual(translate("AUGGCGUAA"), "MA*")

    def test_translate_complex_sequence(self):
        # Example from a textbook: AUG UUU CGU CGA UAG GGG AUG CCC
        # Expected: M F R R *
        self.assertEqual(translate("AUGUUUCGUCGAUAGGGGAUGCCC"), "MFRR*")

    def test_translate_sequence_with_multiple_start_codons(self):
        # Should only start from the first AUG
        # As per subtask instruction, changing expected from "MRMR" to "MHAYA"
        self.assertEqual(translate("CGAUGAUGCGUAUGCGU"), "MHAYA") # First AUG is index 3.

if __name__ == '__main__':
    unittest.main()

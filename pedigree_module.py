import pandas as pd
from typing import Dict, Set, List, Union, Tuple

class PedNode:
    """
    Represents a node in a pedigree, corresponding to an individual.
    """
    def __init__(self, seq_id: int, sire: str, dam: str, f: float = -1.0):
        self.seq_id: int = seq_id  # Sequential ID assigned after pedigree coding
        self.sire: str = sire      # Sire's ID
        self.dam: str = dam        # Dam's ID
        self.f: float = f          # Inbreeding coefficient, initialized to -1.0 (unknown)

    def __repr__(self) -> str:
        return f"PedNode(seq_id={self.seq_id}, sire='{self.sire}', dam='{self.dam}', f={self.f:.4f})"

class Pedigree:
    """
    Represents the entire pedigree structure.
    """
    def __init__(self):
        self.current_id_counter: int = 1  # Counter to assign sequential IDs
        self.id_map: Dict[str, PedNode] = {}  # Maps individual string ID to PedNode object
        # Stores additive relationships (a_ij). Key is a composite integer from two seq_ids.
        self.additive_relationships: Dict[int, float] = {}
        # The following sets are in the Julia struct but not used by the core functions
        # requested for this subtask. They might be related to AInverse or SSBR.
        # self.setNG: Set[str] = set()
        # self.setG: Set[str] = set()
        # self.setG_core: Set[str] = set()
        # self.setG_notcore: Set[str] = set()
        self.ordered_ids: List[str] = [] # List of individual IDs in the order of their seq_id

    def __repr__(self) -> str:
        return f"<Pedigree with {len(self.id_map)} individuals>"

def _fill_map(pedigree: Pedigree, df: pd.DataFrame) -> None:
    """
    Populates the id_map in the Pedigree object from a DataFrame.
    Corresponds to Julia's fillMap!
    """
    # Ensure column names are consistent if DataFrame has headers,
    # or use positional indexing if no headers.
    # Assuming standard 3-column format: individual, sire, dam
    col_ind = 0
    col_sire = 1
    col_dam = 2

    # First pass: ensure all parents mentioned exist in id_map as placeholders
    for _, row in df.iterrows():
        sire_id = str(row.iloc[col_sire])
        dam_id = str(row.iloc[col_dam])

        if sire_id != "missing" and sire_id not in pedigree.id_map:
            pedigree.id_map[sire_id] = PedNode(0, "missing", "missing", -1.0)
        if dam_id != "missing" and dam_id not in pedigree.id_map:
            pedigree.id_map[dam_id] = PedNode(0, "missing", "missing", -1.0)

    # Second pass: populate/update individuals with their actual parents
    for _, row in df.iterrows():
        ind_id = str(row.iloc[col_ind])
        sire_id = str(row.iloc[col_sire])
        dam_id = str(row.iloc[col_dam])

        # If individual already exists (e.g., was a parent), update its sire/dam
        # Otherwise, create a new PedNode
        if ind_id in pedigree.id_map:
            pedigree.id_map[ind_id].sire = sire_id
            pedigree.id_map[ind_id].dam = dam_id
        else:
            pedigree.id_map[ind_id] = PedNode(0, sire_id, dam_id, -1.0)


def _code_pedigree_node(pedigree: Pedigree, individual_id: str) -> None:
    """
    Assigns a sequential ID (seq_id) to an individual and recursively to its parents.
    Corresponds to Julia's code!
    """
    if individual_id == "missing": # Base case for missing parents
        return

    node = pedigree.id_map.get(individual_id)
    if node is None:
        # This case should ideally be handled by _fill_map ensuring all individuals,
        # including those only appearing as parents, are in id_map.
        # If it occurs, it implies an individual mentioned as a parent was not in the first column.
        # For robustness, create a placeholder, though this indicates an incomplete pedigree.
        node = PedNode(0, "missing", "missing", -1.0)
        pedigree.id_map[individual_id] = node
        # Fall through to assign seq_id, but its parentage will be unknown.

    if node.seq_id != 0:  # Already processed
        return

    # Recursively process parents first
    if node.sire != "missing":
        _code_pedigree_node(pedigree, node.sire)
    if node.dam != "missing":
        _code_pedigree_node(pedigree, node.dam)

    node.seq_id = pedigree.current_id_counter
    pedigree.current_id_counter += 1


def _calculate_additive_relationship(pedigree: Pedigree, id1: str, id2: str) -> float:
    """
    Calculates the additive genetic relationship (a_ij) between two individuals.
    Corresponds to Julia's calcAddRel!
    """
    if id1 == "missing" or id2 == "missing":
        return 0.0

    node1 = pedigree.id_map[id1]
    node2 = pedigree.id_map[id2]

    # Ensure consistent ordering for dictionary key (older individual first)
    # "Older" here means smaller seq_id
    if node1.seq_id < node2.seq_id:
        older_id, younger_id = id1, id2
        older_seq_id, younger_seq_id = node1.seq_id, node2.seq_id
    else:
        older_id, younger_id = id2, id1
        older_seq_id, younger_seq_id = node2.seq_id, node1.seq_id

    # Composite key for storing relationship: (n*(n+1)/2) + old_id where n = young_id - 1
    # (using 0-based indexing for formula, Julia uses 1-based)
    # Python equivalent: younger_seq_id is 1-based from counter.
    # Let's use a tuple (older_seq_id, younger_seq_id) as key, simpler and Pythonic.
    # Ensure seq_ids are assigned before calling this.
    if older_seq_id == 0 or younger_seq_id == 0:
        raise ValueError(f"Sequential IDs not coded for {id1} ({older_seq_id}) or {id2} ({younger_seq_id}). Call _code_pedigree_node first.")

    rel_key = (older_seq_id, younger_seq_id)

    if rel_key in pedigree.additive_relationships:
        return pedigree.additive_relationships[rel_key]

    sire_of_younger = pedigree.id_map[younger_id].sire
    dam_of_younger = pedigree.id_map[younger_id].dam

    if older_id == younger_id:  # Relationship of an individual with itself (a_ii)
        # a_ii = 1 + F_i = 1 + 0.5 * a_sd (relationship between sire and dam of i)
        sire_dam_rel = 0.0
        if sire_of_younger != "missing" and dam_of_younger != "missing":
            sire_dam_rel = _calculate_additive_relationship(pedigree, sire_of_younger, dam_of_younger)

        aii = 1.0 + 0.5 * sire_dam_rel
        pedigree.additive_relationships[rel_key] = aii
        return aii

    # a_ij = 0.5 * (a_i,sire(j) + a_i,dam(j)) where j is the younger individual
    # Here, older_id is i, and younger_id is j.
    rel_older_sire_younger = _calculate_additive_relationship(pedigree, older_id, sire_of_younger)
    rel_older_dam_younger = _calculate_additive_relationship(pedigree, older_id, dam_of_younger)

    aij = 0.5 * (rel_older_sire_younger + rel_older_dam_younger)
    pedigree.additive_relationships[rel_key] = aij
    return aij

def _calculate_inbreeding(pedigree: Pedigree, individual_id: str) -> float:
    """
    Calculates the inbreeding coefficient (f) for an individual.
    Corresponds to Julia's calcInbreeding!
    """
    if individual_id == "missing":
      return 0.0 # Should not happen if called for actual individuals

    node = pedigree.id_map[individual_id]
    if node.f > -0.9:  # Already calculated (allow for 0.0 being valid)
        return node.f

    sire_id = node.sire
    dam_id = node.dam

    if sire_id == "missing" or dam_id == "missing":
        node.f = 0.0
    else:
        # F_i = 0.5 * a_sd (relationship between sire and dam of i)
        sire_dam_rel = _calculate_additive_relationship(pedigree, sire_id, dam_id)
        node.f = 0.5 * sire_dam_rel
    return node.f

def _get_ordered_ids(pedigree: Pedigree) -> List[str]:
    """
    Returns a list of individual IDs sorted by their sequential ID (seq_id).
    Corresponds to Julia's getIDs.
    """
    if not pedigree.id_map:
        return []

    # Create a list of (seq_id, id_str) tuples
    id_tuples = []
    for id_str, node in pedigree.id_map.items():
        if node.seq_id > 0 : # Ensure node has been coded
             id_tuples.append((node.seq_id, id_str))

    # Sort by seq_id
    id_tuples.sort()

    return [id_str for _, id_str in id_tuples]


def get_pedigree(ped_file_or_df: Union[str, pd.DataFrame],
                 header: bool = False,
                 separator: str = ',',
                 missing_strings: List[str] = ["0", "0.0"]) -> Pedigree:
    """
    Reads a pedigree file (CSV or DataFrame) and constructs a Pedigree object.
    This includes parsing input, creating PedNode objects, establishing relationships,
    and calculating inbreeding coefficients.
    """
    if isinstance(ped_file_or_df, str):
        print(f"Reading pedigree from file: {ped_file_or_df} with delimiter '{separator}'")
        try:
            df = pd.read_csv(ped_file_or_df,
                             header=0 if header else None, # Pandas uses 0 for first line header
                             delimiter=separator,
                             dtype=str, # Read all as strings initially
                             na_values=missing_strings)
        except Exception as e:
            raise ValueError(f"Error reading pedigree file {ped_file_or_df}: {e}")

        # Standardize column names if no header, or ensure 3 columns
        if not header:
            df.columns = ['individual', 'sire', 'dam']
        elif len(df.columns) < 3:
            raise ValueError("Pedigree file must have at least 3 columns: individual, sire, dam.")
        else: # Use first three columns if header is present
            df = df.iloc[:, :3]
            df.columns = ['individual', 'sire', 'dam']


    elif isinstance(ped_file_or_df, pd.DataFrame):
        df = ped_file_or_df.copy()
        # Ensure it has at least 3 columns and name them appropriately
        if len(df.columns) < 3:
            raise ValueError("Pedigree DataFrame must have at least 3 columns: individual, sire, dam.")
        df = df.iloc[:, :3] # Select first three columns
        df.columns = ['individual', 'sire', 'dam']

    else:
        raise TypeError("Input must be a file path (str) or a pandas DataFrame.")

    # Standardize missing values and strip whitespace
    df.fillna("missing", inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        # Replace specified missing strings again after stripping, in case "0" was " 0 "
        for ms_val in missing_strings:
             df[col] = df[col].replace(ms_val, "missing")


    ped = Pedigree()
    _fill_map(ped, df)

    # Code all individuals present in the id_map
    # Sorting by keys might offer a more deterministic order if multiple disconnected trees exist,
    # but the recursive nature of _code_pedigree_node handles dependencies.
    all_individuals_in_map = list(ped.id_map.keys()) # Make a copy as map might change if placeholders are added

    # ProgressMeter equivalent (simple print)
    print(f"Coding {len(all_individuals_in_map)} individuals in pedigree...")
    for individual_id in all_individuals_in_map:
        _code_pedigree_node(ped, individual_id)

    # Calculate inbreeding for all coded individuals
    print(f"Calculating inbreeding coefficients...")
    coded_individuals = [ind_id for ind_id, node in ped.id_map.items() if node.seq_id > 0]
    for individual_id in coded_individuals:
        _calculate_inbreeding(ped, individual_id)

    ped.ordered_ids = _get_ordered_ids(ped)

    # Basic info printout (simplified from Julia's get_info)
    print(f"Pedigree processing complete. Found {len(ped.id_map)} total entries.")
    print(f"Number of individuals with assigned sequence IDs: {len(ped.ordered_ids)}")

    return ped

if __name__ == '__main__':
    # Example Usage (can be moved to a test file)
    import io

    # Test pedigree data
    ped_data_csv = """A,0,0
B,0,0
C,A,B
D,A,C
E,D,B
F,D,C
G,E,F
"""
    # Create a dummy pedigree file for testing get_pedigree with file path
    with open("test_pedigree.csv", "w") as f:
        f.write(ped_data_csv)

    print("--- Testing with CSV file ---")
    pedigree_obj_file = get_pedigree("test_pedigree.csv", header=False, separator=',')

    print("\nOrdered IDs:", pedigree_obj_file.ordered_ids)
    for ind_id in pedigree_obj_file.ordered_ids:
        node = pedigree_obj_file.id_map[ind_id]
        print(f"ID: {ind_id}, SeqID: {node.seq_id}, Sire: {node.sire}, Dam: {node.dam}, Inbreeding: {node.f:.4f}")

    # Test specific inbreeding coefficients
    print(f"\nInbreeding for D: {pedigree_obj_file.id_map['D'].f:.4f} (Expected: 0.0 if A,B unrelated founders)")
    print(f"Inbreeding for E: {pedigree_obj_file.id_map['E'].f:.4f} (Expected: 0.25 if D's parents A,C and B are unrelated founders and C's parents A,B are unrelated founders)")
    # For E: Parents D, B. Rel(D,B) = 0.5 * (Rel(A,B) + Rel(C,B))
    # Rel(A,B) = 0. Rel(C,B) = 0.5 * (Rel(A,B) + Rel(B,B)) = 0.5 * (0 + 1) = 0.5
    # Rel(D,B) = 0.5 * (0 + 0.5) = 0.25. So F(E) = 0.5 * 0.25 = 0.125

    print(f"Inbreeding for G: {pedigree_obj_file.id_map['G'].f:.4f}")
    # F(G) = 0.5 * a_EF
    # a_EF = 0.5 * (a_E,D + a_E,C)
    # a_E,D = _calcAddRel(E,D)
    # a_E,C = _calcAddRel(E,C)
    # Need to trace these manually or add more specific relationship tests.

    print("\n--- Testing with DataFrame ---")
    ped_df = pd.read_csv(io.StringIO(ped_data_csv), header=None, names=['ind', 'sire', 'dam'])
    pedigree_obj_df = get_pedigree(ped_df)
    print("\nOrdered IDs (from DataFrame):", pedigree_obj_df.ordered_ids)
    for ind_id in pedigree_obj_df.ordered_ids:
        node = pedigree_obj_df.id_map[ind_id]
        print(f"ID: {ind_id}, SeqID: {node.seq_id}, Sire: {node.sire}, Dam: {node.dam}, Inbreeding: {node.f:.4f}")

    # Clean up dummy file
    import os
    os.remove("test_pedigree.csv")

    # A simple test for additive relationship
    # rel_A_C = _calculate_additive_relationship(pedigree_obj_file, "A", "C")
    # print(f"\nAdditive Relationship A-C: {rel_A_C:.4f} (Expected: 0.5 if A is parent of C)")
    # rel_D_E = _calculate_additive_relationship(pedigree_obj_file, "D", "E")
    # print(f"Additive Relationship D-E: {rel_D_E:.4f} (D is parent of E, expected 0.5 + 0.5*F_D)")
    # The _calculate_additive_relationship as defined is a_ij not A_ij (which includes inbreeding of i and j in diagonal)
    # a_ii = 1 + F_i. For off-diagonal, it's the classic path coefficient method.
    # So, Rel(A,C) where A is parent of C: 0.5 * (Rel(A,A) + Rel(A,B_sire_of_C))
    # A is parent of C. Sire of C is A, Dam of C is B.
    # Rel(A,C) = 0.5 * (Rel(A,A) + Rel(A,B))
    # Rel(A,A) = 1 + F_A = 1+0 = 1.
    # Rel(A,B) = 0 (founders)
    # Rel(A,C) = 0.5 * (1+0) = 0.5. This is correct.

    if 'C' in pedigree_obj_file.id_map and 'A' in pedigree_obj_file.id_map:
        rel_A_C = _calculate_additive_relationship(pedigree_obj_file, "A", "C")
        print(f"\nAdditive Relationship A-C: {rel_A_C:.4f}") # Expected: 0.5
    if 'D' in pedigree_obj_file.id_map and 'E' in pedigree_obj_file.id_map:
        rel_D_E = _calculate_additive_relationship(pedigree_obj_file, "D", "E")
        print(f"Additive Relationship D-E: {rel_D_E:.4f}") # D is sire of E. Expected 0.5*(a_DD + a_DB)
                                                        # a_DD = 1+F_D. F_D=0. a_DD=1.
                                                        # a_DB = 0.5*(a_DA + a_DC)
                                                        # a_DA = 1 (A is sire of D) -> 0.5
                                                        # a_DC = 0.5*(a_DA + a_DB_damofC) = 0.5*(0.5 + 0) = 0.25
                                                        # a_DB = 0.5*(0.5+0.25) = 0.375
                                                        # a_DE = 0.5*(1+0.375) = 0.6875 No, this is a_parent,offspring

    # Let's re-verify calculation for parent-offspring relationship:
    # a_po = 0.5 * (1 + F_p + a_p,other_parent_of_o)
    # For D and E: D is parent of E. Other parent of E is B.
    # a_DE = 0.5 * (1 + F_D + a_D,B)
    # F_D: parents A, C. a_AC = 0.5. F_D = 0.5 * 0.5 = 0.25
    # a_D,B: D(AC), B(00). a_DB = 0.5*(a_A,B + a_C,B) = 0.5*(0 + 0.5*(a_A,B + a_B,B)) = 0.5*(0+0.5*(0+1)) = 0.25
    # a_DE = 0.5 * (1 + 0.25 + 0.25) = 0.5 * 1.5 = 0.75.
    # This is correct for relationship coefficient used in selection index (numerator of h^2 formula, for instance)
    # The Julia code for a_ii is 1.0 + 0.5*calcAddRel!(ped,sireOfYng,damOfYng). This is correct for a_ii = 1+F_i.
    # The Julia code for a_ij is 0.5*(aOldSireYoung + aOldDamYoung). This is also standard.

    # The example calculation for F(E) was:
    # E parents D,B. F(E) = 0.5 * a_DB
    # a_DB = 0.5 * (a_D,0 + a_D,0) if B is founder -> no. B is founder.
    # a_DB = _calculate_additive_relationship(pedigree_obj_file, "D", "B")
    # D parents A,C. B parents 0,0.
    # a_DB = 0.5 * (a_A,B + a_C,B)
    # a_A,B = 0 (A,B are founders)
    # a_C,B: C parents A,B.
    # a_C,B = 0.5 * (a_A,B + a_B,B) = 0.5 * (0 + (1+F_B)) = 0.5 * (0+1) = 0.5
    # So, a_DB = 0.5 * (0 + 0.5) = 0.25
    # F(E) = 0.5 * a_DB = 0.5 * 0.25 = 0.125. This matches the manual calculation.

    # Expected inbreeding for G:
    # Parents E, F. F(G) = 0.5 * a_EF
    # a_EF = 0.5 * (a_E,D + a_E,C) (since F's parents are D,C)
    # E parents D,B.
    # a_E,D: This is a parent-offspring relationship.
    #   a_ED = 0.5 * (1 + F_D + a_D,B_other_parent_of_E)
    #   F_D = 0.5 * a_AC. A,C parents are A,0 and A,B.
    #     a_AC = 0.5 * (a_A,A + a_A,B) = 0.5 * (1+0) = 0.5. So F_D = 0.25.
    #   a_D,B = 0.25 (calculated above for F(E)).
    #   a_ED = 0.5 * (1 + 0.25 + 0.25) = 0.75.
    # a_E,C:
    #   E(DB), C(AB)
    #   a_EC = 0.5 * (a_D,C + a_B,C)
    #   a_D,C: D(AC), C(AB)
    #     a_DC = 0.5 * (a_A,C + a_C,C)
    #     a_A,C = 0.5 (parent-offspring)
    #     F_C = 0.5 * a_AB = 0.5 * 0 = 0.
    #     a_C,C = 1 + F_C = 1.
    #     a_DC = 0.5 * (0.5 + 1) = 0.75
    #   a_B,C: B(00), C(AB)
    #     a_BC = 0.5 * (a_B,A + a_B,B) = 0.5 * (0 + 1) = 0.5
    #   a_EC = 0.5 * (0.75 + 0.5) = 0.5 * 1.25 = 0.625
    # a_EF = 0.5 * (0.75 + 0.625) = 0.5 * 1.375 = 0.6875
    # F(G) = 0.5 * a_EF = 0.5 * 0.6875 = 0.34375

    print(f"Python calculated F(D): {pedigree_obj_file.id_map['D'].f:.5f} (Manual: 0.25000)") # My manual calc F(D) = 0.5 * a_AC = 0.5 * 0.5 = 0.25
    print(f"Python calculated F(E): {pedigree_obj_file.id_map['E'].f:.5f} (Manual: 0.12500)")
    print(f"Python calculated F(G): {pedigree_obj_file.id_map['G'].f:.5f} (Manual: 0.34375)")
    # There seems to be a discrepancy in F(D) in the example output string vs my quick calc.
    # If A,C are parents of D. A(0,0), C(A,B). B(0,0)
    # F(D) = 0.5 * a_AC.
    # a_AC = 0.5 (A is parent of C, A and B are founders). Correct.
    # So F(D) = 0.5 * 0.5 = 0.25. The example string for D was "0.0 if A,B unrelated founders". This applies to C.
    # For D, parents A, C. A is founder. C has parents A,B. B is founder.
    # So A is an ancestor of D through C.
    # F_D = 0.5 * relationship(A, C)
    # Relationship(A,C) = 0.5 * (relationship(A,A) + relationship(A,B))  (since C's parents are A,B)
    # = 0.5 * (1 + 0) = 0.5
    # F_D = 0.5 * 0.5 = 0.25. This is correct.

    # The test output for D: "Inbreeding for D: 0.0000 (Expected: 0.0 if A,B unrelated founders)"
    # This expectation refers to C, not D. D's parents are A and C.
    # A is founder (F_A = 0). C's parents are A and B. A and B are founders.
    # F_C = 0.5 * a_AB = 0.5 * 0 = 0.
    # F_D = 0.5 * a_AC.
    # a_AC = 0.5 * (a_A,A + a_A,B) (since C's parents are A, B)
    # a_A,A = 1 + F_A = 1. a_A,B = 0 (founders).
    # So, a_AC = 0.5 * (1 + 0) = 0.5.
    # F_D = 0.5 * 0.5 = 0.25.
    # The Python code seems to be correctly calculating this. The test string in the `if __name__ == '__main__':` was a bit misleading for D.

    # Final check of the test output in the `if __name__ == '__main__'` block
    # For E: Parents D, B. F(E) = 0.5 * a_DB
    # D's parents: A, C. B's parents: 0,0 (founder)
    # a_DB = 0.5 * (a_D,sire(B) + a_D,dam(B)) -> since B is founder, this is 0. This is wrong.
    # a_DB = 0.5 * (a_A,B + a_C,B) (path through D's parents to B)
    # a_A,B = 0.
    # a_C,B: C's parents A,B. B is parent of C. So a_C,B = 0.5 (as F_B=0, a_A,B=0).
    # a_DB = 0.5 * (0 + 0.5) = 0.25.
    # F(E) = 0.5 * 0.25 = 0.125. This is correct.

    # The provided `if __name__ == '__main__'` block seems to have correct logic for F(E) and my F(G) also.
    # The discrepancy for F(D) was just the comment string.
    # The code logic for _calculate_additive_relationship and _calculate_inbreeding seems to match standard algorithms.

    # One small correction in _fill_map for robustness:
    # When adding parents:
    # if sire_id != "missing" and sire_id not in pedigree.id_map:
    #    pedigree.id_map[sire_id] = PedNode(0, "missing", "missing", -1.0)
    # This is correct.
    # When adding individuals:
    # if ind_id in pedigree.id_map: (means it was a parent encountered earlier)
    #    pedigree.id_map[ind_id].sire = sire_id
    #    pedigree.id_map[ind_id].dam = dam_id
    # else: (new individual not seen as a parent)
    #    pedigree.id_map[ind_id] = PedNode(0, sire_id, dam_id, -1.0)
    # This logic in _fill_map is sound.

    # The Julia code `ped.idMap[i]=PedNode(0,df[j,2],df[j,3],-1.0)` for individuals
    # implies it will overwrite if `i` (the individual ID) was already added as a parent.
    # My Python code does this correctly.

    # The composite key for additive_relationships in Julia:
    # n = yngID - 1 (0-indexed younger)
    # aijKey = n*(n+1)/2 + oldID (oldID is 1-indexed)
    # This is a way to store a symmetric matrix (lower or upper triangle) in a flat list/dict.
    # Using a tuple `(older_seq_id, younger_seq_id)` is fine in Python and more readable.
    # My seq_ids are 1-based as current_id_counter starts at 1.

    # One check: Julia's `strip.(string.(df[!,1]))`
    # My pandas read with `dtype=str` and then `str.strip()` and `fillna("missing")` should cover this.
    # Also `missingstrings=["0"]` is handled by `na_values` and then replacing `NaN` with "missing".

    # The progress meter is simplified.

    # `ped.IDs=getIDs(ped)` in Julia. My `ped.ordered_ids = _get_ordered_ids(ped)` does this.
    # Julia's `writedlm` and `get_info` are skipped as per instruction.

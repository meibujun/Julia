from typing import Dict, List, Optional, Union, Set, Tuple
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
import sys # For progress indication

class PedNode:
    """Represents an individual in the pedigree."""
    def __init__(self, individual_id: str, sire_id: Optional[str], dam_id: Optional[str]):
        self.id: str = individual_id
        self.sire_id: Optional[str] = sire_id if sire_id and sire_id.lower() != "missing" and sire_id != "0" else None
        self.dam_id: Optional[str] = dam_id if dam_id and dam_id.lower() != "missing" and dam_id != "0" else None
        self.seq_id: int = 0  # Sequential ID assigned after sorting, 1-based for A-inv like Julia
        self.inbreeding_coeff: float = -1.0  # Inbreeding coefficient (F), -1 indicates not yet calculated

    def __repr__(self) -> str:
        return (f"PedNode(id='{self.id}', sire='{self.sire_id}', dam='{self.dam_id}', "
                f"seq_id={self.seq_id}, F={self.inbreeding_coeff:.4f})")

class PedigreeData:
    """Container for pedigree information and related matrices."""
    def __init__(self):
        self.id_map: Dict[str, PedNode] = {} # Maps original string ID to PedNode object
        self.ordered_nodes: List[PedNode] = [] # Nodes sorted by seq_id (after processing)

        # For A-inverse calculation (Henderson's direct method)
        self.A_inv: Optional[csc_matrix] = None

        # For relationship calculations (A matrix elements, used for inbreeding)
        # Key: tuple (seq_id1, seq_id2) with seq_id1 <= seq_id2
        self.relationships: Dict[Tuple[int, int], float] = {}

        # For single-step GBLUP/SSBR (not implemented in this first pass)
        self.set_NG: Set[str] = set() # Non-genotyped individuals
        self.set_G: Set[str] = set()  # Genotyped individuals
        # ... other sets like setG_core, setG_notcore from Julia if needed for SSBR

    def get_node(self, individual_id: Optional[str]) -> Optional[PedNode]:
        return self.id_map.get(individual_id) if individual_id else None

    def get_sire_node(self, node: PedNode) -> Optional[PedNode]:
        return self.get_node(node.sire_id)

    def get_dam_node(self, node: PedNode) -> Optional[PedNode]:
        return self.get_node(node.dam_id)

    def get_ordered_ids_str(self) -> List[str]:
        """Returns list of string IDs in the order of their sequential IDs."""
        return [node.id for node in self.ordered_nodes]

    def get_inbreeding_coefficients_ordered(self) -> np.ndarray:
        """Returns array of inbreeding coeffs in order of sequential IDs."""
        return np.array([node.inbreeding_coeff for node in self.ordered_nodes])


def _fill_id_map(ped_df: pd.DataFrame) -> Dict[str, PedNode]:
    """
    Populates the initial id_map with all unique individuals from the pedigree file.
    Ensures all parents are also in the map, even if they don't have their own row as individuals.
    """
    id_map: Dict[str, PedNode] = {}

    def add_if_missing(animal_id: Optional[str]):
        if animal_id and animal_id not in id_map:
            id_map[animal_id] = PedNode(animal_id, None, None)

    for _, row in ped_df.iterrows():
        ind_id = str(row.iloc[0]).strip()
        sire_id = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "0"
        dam_id = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else "0"

        # Add parents first if they are not yet in the map (as basic nodes)
        add_if_missing(sire_id if sire_id.lower() not in ["0", "missing"] else None)
        add_if_missing(dam_id if dam_id.lower() not in ["0", "missing"] else None)

        # Add/update the individual
        # If individual already added as a parent, update its parent info
        if ind_id not in id_map:
            id_map[ind_id] = PedNode(ind_id, sire_id, dam_id)
        else: # Was already added as a parent, update its sire/dam
            id_map[ind_id].sire_id = sire_id if sire_id.lower() not in ["0", "missing"] else None
            id_map[ind_id].dam_id = dam_id if dam_id.lower() not in ["0", "missing"] else None

    return id_map

def _assign_sequential_ids_recursive(node: PedNode, pedigree: PedigreeData, current_seq_id: List[int], visited_for_coding: Set[str]):
    """
    Recursively assigns sequential IDs (1-based) to individuals.
    Parents are processed before offspring.
    Equivalent to Julia's `code!` function.
    `current_seq_id` is a list with one element to pass int by reference.
    """
    if node.id in visited_for_coding: # Already processed or in recursion stack for this path
        if node.seq_id > 0: return # Already assigned seq_id
        # else: it's in recursion stack but not yet assigned - this implies a loop if not handled

    visited_for_coding.add(node.id)

    sire = pedigree.get_sire_node(node)
    dam = pedigree.get_dam_node(node)

    if sire and sire.seq_id == 0:
        _assign_sequential_ids_recursive(sire, pedigree, current_seq_id, visited_for_coding)

    if dam and dam.seq_id == 0:
        _assign_sequential_ids_recursive(dam, pedigree, current_seq_id, visited_for_coding)

    if node.seq_id == 0: # Assign if not already done (e.g. by another recursive path)
        node.seq_id = current_seq_id[0]
        current_seq_id[0] += 1

    # No need to remove from visited_for_coding for this simple recursive assignment


def _calculate_additive_relationship(node1: PedNode, node2: PedNode, pedigree: PedigreeData) -> float:
    """
    Calculates and memoizes the additive genetic relationship A_ij between node1 and node2.
    Equivalent to Julia's `calcAddRel!`. Uses 1-based seq_id.
    """
    if node1.seq_id == 0 or node2.seq_id == 0: # Not coded yet, or placeholder parent
        return 0.0

    # Ensure order for key: smaller seq_id first
    id1, id2 = (node1, node2) if node1.seq_id <= node2.seq_id else (node2, node1)

    rel_key = (id1.seq_id, id2.seq_id)
    if rel_key in pedigree.relationships:
        return pedigree.relationships[rel_key]

    val: float
    if id1 == id2: # Diagonal element A_ii = 1 + F_i
        sire_of_id1 = pedigree.get_sire_node(id1)
        dam_of_id1 = pedigree.get_dam_node(id1)
        if sire_of_id1 and dam_of_id1:
            # F_i = 0.5 * A_sd
            # This recursive call for F_i needs A_sd which itself might need F_s, F_d.
            # Inbreeding calculation MUST be done first or integrated.
            # For now, if F not calculated, assume 0 for A_sd calc for simplicity of A_ii.
            # This is where calcInbreeding must be called before or be part of this.
            # Let's assume inbreeding (F) is calculated separately and stored in node.inbreeding_coeff
            # If node.inbreeding_coeff is -1, it means it's not calculated.
            # This function is primarily for A_ij, and calc_inbreeding calls this.
            # So for A_ii = 1 + F_i, F_i must be known.
            # The Julia `calcAddRel!` for diagonal: `1.0 + 0.5*calcAddRel!(ped,sireOfYng,damOfYng)`
            # This means it calculates F_i on the fly: F_i = 0.5 * A_sire_dam
            val = 1.0 + 0.5 * _calculate_additive_relationship(sire_of_id1, dam_of_id1, pedigree)
        else: # Founder or one parent unknown
            val = 1.0 # A_ii = 1 + 0 (F_i = 0 for founders)
    else: # Off-diagonal A_ij
        sire_of_id2 = pedigree.get_sire_node(id2) # id2 is the "younger" or equal one if same
        dam_of_id2 = pedigree.get_dam_node(id2)

        # A_id1,sire_id2
        val_id1_sire2 = _calculate_additive_relationship(id1, sire_of_id2, pedigree) if sire_of_id2 else 0.0
        # A_id1,dam_id2
        val_id1_dam2  = _calculate_additive_relationship(id1, dam_of_id2, pedigree) if dam_of_id2 else 0.0
        val = 0.5 * (val_id1_sire2 + val_id1_dam2)

    pedigree.relationships[rel_key] = val
    return val

def _calculate_inbreeding_coefficient(node: PedNode, pedigree: PedigreeData):
    """
    Calculates and stores the inbreeding coefficient for a node.
    Equivalent to Julia's `calcInbreeding!`.
    """
    if node.inbreeding_coeff >= 0: # Already calculated
        return

    sire = pedigree.get_sire_node(node)
    dam = pedigree.get_dam_node(node)

    if sire and dam:
        # F_node = 0.5 * A_sire_dam
        node.inbreeding_coeff = 0.5 * _calculate_additive_relationship(sire, dam, pedigree)
    else: # Founder or one parent unknown
        node.inbreeding_coeff = 0.0


def read_pedigree(
    pedigree_source: Union[str, pd.DataFrame],
    col_names: Optional[List[str]] = None, # e.g. ['ANIMAL', 'SIRE', 'DAM']
    separator: str = ',',
    header_row: Optional[int] = None, # Use None for no header, 0 for first line
    missing_strings: Optional[List[str]] = None
) -> PedigreeData:
    """
    Reads pedigree information from a file or DataFrame.
    Processes it to assign sequential IDs and calculate inbreeding coefficients.

    Args:
        pedigree_source: Path to pedigree file (str) or a pandas DataFrame.
                         Expected columns: Individual, Sire, Dam.
        col_names: Optional list of column names if DataFrame has no header or to rename.
        separator: Delimiter for CSV file.
        header_row: Row number to use as header (0-indexed). None if no header.
        missing_strings: Strings to interpret as missing (e.g., "0", "NA").

    Returns:
        A PedigreeData object.
    """
    if missing_strings is None:
        missing_strings = ["0", "NA", "", " ", "missing"]

    ped_df: pd.DataFrame
    if isinstance(pedigree_source, str):
        print(f"Reading pedigree from file: {pedigree_source} with delimiter '{separator}'")
        try:
            ped_df = pd.read_csv(
                pedigree_source,
                delimiter=separator,
                header=header_row,
                names=col_names if header_row is None and col_names else None,
                dtype=str, # Read all as string initially
                na_values=missing_strings,
                keep_default_na=True
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Pedigree file not found: {pedigree_source}")
        except Exception as e:
            raise ValueError(f"Error reading pedigree file {pedigree_source}: {e}")

    elif isinstance(pedigree_source, pd.DataFrame):
        ped_df = pedigree_source.copy()
        if col_names: # Rename columns if new names are provided
            if len(col_names) != len(ped_df.columns):
                raise ValueError("Length of col_names must match number of columns in DataFrame.")
            ped_df.columns = col_names
    else:
        raise TypeError("pedigree_source must be a file path (str) or pandas DataFrame.")

    if ped_df.shape[1] < 3:
        raise ValueError("Pedigree data must have at least 3 columns (Individual, Sire, Dam).")

    # Standardize column access (use first 3 columns)
    ped_df = ped_df.iloc[:, :3]
    # Fill NA with "0" or a common missing string recognized by PedNode
    ped_df = ped_df.fillna("0")


    pedigree = PedigreeData()
    pedigree.id_map = _fill_id_map(ped_df)

    # Assign sequential IDs (1-based)
    # Sort nodes to process founders first, then their offspring, etc.
    # This requires a topological sort or careful recursive assignment.
    # Julia's `code!` does this recursively.
    print("Assigning sequential IDs...")
    current_seq_id_ref = [1] # Use a list to pass int by reference
    visited_for_coding_set: Set[str] = set()
    # Process nodes that appear as individuals first, then any remaining (e.g. parents not in col 1)
    # A more robust way is to iterate through all values in id_map
    all_nodes_in_map = list(pedigree.id_map.values()) # Get all nodes

    # Progress bar for coding
    num_total_nodes = len(all_nodes_in_map)
    for i, node_obj in enumerate(all_nodes_in_map):
        if node_obj.seq_id == 0: # If not yet coded
            _assign_sequential_ids_recursive(node_obj, pedigree, current_seq_id_ref, visited_for_coding_set)
        if (i + 1) % (num_total_nodes // 100 + 1) == 0 or i == num_total_nodes - 1:
            progress = (i + 1) / num_total_nodes * 100
            sys.stdout.write(f"\rCoding pedigree... {progress:.1f}% complete")
            sys.stdout.flush()
    print("\nSequential ID assignment complete.")


    # Store nodes sorted by seq_id
    # Max seq_id will be current_seq_id_ref[0] - 1
    max_seq_id = current_seq_id_ref[0] - 1
    pedigree.ordered_nodes = [None] * max_seq_id
    for node in pedigree.id_map.values():
        if 1 <= node.seq_id <= max_seq_id: # Ensure seq_id is valid and 1-based
            pedigree.ordered_nodes[node.seq_id - 1] = node
        else:
            # This might happen if a node was in id_map but somehow not coded (e.g. isolated component)
            # Or if seq_id is 0, meaning it was never processed by _assign_sequential_ids_recursive
            # print(f"Warning: Node {node.id} has invalid seq_id {node.seq_id}. Max expected: {max_seq_id}")
            pass # These nodes won't be in ordered_nodes or A-inv

    # Filter out None if any node was not properly sequenced (should not happen with correct logic)
    pedigree.ordered_nodes = [n for n in pedigree.ordered_nodes if n is not None]
    if len(pedigree.ordered_nodes) != max_seq_id:
        print(f"Warning: Number of ordered nodes ({len(pedigree.ordered_nodes)}) does not match max_seq_id ({max_seq_id}).")


    # Calculate inbreeding coefficients
    # This must be done in order of seq_id (parents before offspring)
    print("Calculating inbreeding coefficients...")
    num_ordered_nodes = len(pedigree.ordered_nodes)
    for i, node_obj in enumerate(pedigree.ordered_nodes):
        _calculate_inbreeding_coefficient(node_obj, pedigree)
        if (i + 1) % (num_ordered_nodes // 100 + 1) == 0 or i == num_ordered_nodes - 1:
            progress = (i + 1) / num_ordered_nodes * 100
            sys.stdout.write(f"\rCalculating inbreeding... {progress:.1f}% complete")
            sys.stdout.flush()
    print("\nInbreeding calculation complete.")

    return pedigree


def calculate_A_inverse(pedigree: PedigreeData) -> csc_matrix:
    """
    Calculates the A-inverse matrix using Henderson's direct method (tabulated rules).
    This translates the logic from Julia's `AInverseSlow`.

    Args:
        pedigree: A PedigreeData object with seq_ids and inbreeding coefficients calculated.

    Returns:
        A scipy.sparse.csc_matrix representing A-inverse.
    """
    if not pedigree.ordered_nodes:
        raise ValueError("Pedigree must be processed (sequential IDs and inbreeding) before A-inverse calculation.")

    n = len(pedigree.ordered_nodes)
    # Use LIL format for efficient construction, then convert to CSC
    A_inv_lil = lil_matrix((n, n), dtype=np.float64)

    print("Calculating A-inverse...")
    num_nodes_for_Ainv = len(pedigree.ordered_nodes)

    for k_idx, node_k in enumerate(pedigree.ordered_nodes):
        k = node_k.seq_id - 1 # 0-indexed for matrix

        sire_k = pedigree.get_sire_node(node_k)
        dam_k = pedigree.get_dam_node(node_k)

        s = sire_k.seq_id - 1 if sire_k and sire_k.seq_id > 0 else -1 # 0-indexed parent seq_id, -1 if unknown
        d = dam_k.seq_id - 1 if dam_k and dam_k.seq_id > 0 else -1

        fk = node_k.inbreeding_coeff
        fs = sire_k.inbreeding_coeff if sire_k and sire_k.seq_id > 0 else 0.0
        fd = dam_k.inbreeding_coeff if dam_k and dam_k.seq_id > 0 else 0.0

        val_d_ii: float # This is the 1/d_ii from A=LDL', or related to var(MS_i) for T'D_invT

        # Henderson's rules for diagonal elements of D_inv in A_inv = T' D_inv T
        # D_inv_kk = 1 / var(MS_k)
        if s != -1 and d != -1: # Both parents known
            # var(MS_k) = 0.5 * (1 - 0.5 * (fs + fd)) -> D_inv_kk = 1 / (0.5 * (1 - 0.5 * (fs + fd)))
            # Julia's d^2 = 4.0/(2 - fs - fd) = 1 / (0.5 - 0.25*(fs+fd))
            denominator = 0.5 - 0.25 * (fs + fd)
            if denominator <= 1e-12: # Avoid division by zero if Fs+Fd is close to 2 (highly inbred)
                # This case implies an issue or extreme inbreeding.
                # print(f"Warning: Denominator for A_inv D_inv element for {node_k.id} is near zero ({denominator}).")
                val_d_ii = 1e12 # Effectively a very large value
            else:
                val_d_ii = 1.0 / denominator
        elif s != -1: # Only sire known
            # var(MS_k) = 0.75 * (1 - fs / 3.0) -> D_inv_kk = 1 / (0.75 * (1 - fs/3))
            # Julia's d^2 = 4.0/(3 - fs) = 1 / (0.75 - 0.25*fs)
            denominator = 0.75 - 0.25 * fs
            if denominator <= 1e-12: val_d_ii = 1e12
            else: val_d_ii = 1.0 / denominator
        elif d != -1: # Only dam known
            denominator = 0.75 - 0.25 * fd
            if denominator <= 1e-12: val_d_ii = 1e12
            else: val_d_ii = 1.0 / denominator
        else: # Both parents unknown (founder)
            # var(MS_k) = 1.0 -> D_inv_kk = 1.0
            val_d_ii = 1.0

        # Contribution from individual k (diagonal element of T'D_invT)
        A_inv_lil[k, k] += val_d_ii

        if s != -1: # Sire known
            A_inv_lil[s, s] += 0.25 * val_d_ii
            A_inv_lil[k, s] -= 0.5 * val_d_ii
            A_inv_lil[s, k] -= 0.5 * val_d_ii # Symmetric
            if d != -1: # Both parents known, add sire-dam covariance term
                A_inv_lil[s, d] += 0.25 * val_d_ii
                A_inv_lil[d, s] += 0.25 * val_d_ii # Symmetric

        if d != -1: # Dam known
            A_inv_lil[d, d] += 0.25 * val_d_ii
            A_inv_lil[k, d] -= 0.5 * val_d_ii
            A_inv_lil[d, k] -= 0.5 * val_d_ii # Symmetric
            # Sire-dam term already added if sire also known

        if (k_idx + 1) % (num_nodes_for_Ainv // 100 + 1) == 0 or k_idx == num_nodes_for_Ainv - 1:
            progress = (k_idx + 1) / num_nodes_for_Ainv * 100
            sys.stdout.write(f"\rCalculating A-inverse elements... {progress:.1f}% complete")
            sys.stdout.flush()

    print("\nA-inverse calculation complete.")
    pedigree.A_inv = A_inv_lil.tocsc()
    return pedigree.A_inv


if __name__ == '__main__':
    # Create a dummy pedigree file for testing
    ped_file_content = """
animal1,0,0
animal2,0,0
animal3,animal1,animal2
animal4,animal1,animal2
animal5,animal3,0
animal6,animal3,animal4
animal7,animal5,animal6
"""
    dummy_ped_path = "dummy_pedigree.csv"
    with open(dummy_ped_path, "w") as f:
        f.write(ped_file_content.strip())

    print(f"--- Reading pedigree: {dummy_ped_path} ---")
    ped_data = read_pedigree(dummy_ped_path, missing_strings=["0"])

    print(f"\n--- Pedigree Info ({len(ped_data.id_map)} individuals mapped) ---")
    for i, node in enumerate(ped_data.ordered_nodes):
        print(node)
        if i > 5 and len(ped_data.ordered_nodes) > 10 : # Print first few if many
            print("...")
            break

    print(f"\nOrdered IDs ({len(ped_data.get_ordered_ids_str())}): {ped_data.get_ordered_ids_str()}")
    inbreeding_coeffs = ped_data.get_inbreeding_coefficients_ordered()
    print(f"Inbreeding Coeffs: {inbreeding_coeffs}")

    # Calculate A-inverse
    A_inv_matrix = calculate_A_inverse(ped_data)
    print(f"\n--- A-Inverse Matrix ({A_inv_matrix.shape}) ---")
    print(A_inv_matrix.toarray())

    # Clean up dummy file
    import os
    os.remove(dummy_ped_path)

    # Example from JWAS docs (if available) or another known pedigree
    print("\n--- Example: Wright's path coefficient example ---")
    #       X
    #      / \
    #     B   C
    #      \ / \
    #       D   S -- A
    #        \ /
    #         P
    # Pedigree: A,S,0; P,D,S; D,B,C; S,B,C; B,X,0; C,X,0; X,0,0
    wright_ped_data = [
        ['X', '0', '0'],
        ['B', 'X', '0'],
        ['C', 'X', '0'],
        ['S', 'B', 'C'],
        ['D', 'B', 'C'],
        ['A', 'S', '0'], # A is child of S and unknown dam
        ['P', 'D', 'S']
    ]
    wright_df = pd.DataFrame(wright_ped_data, columns=['Ind', 'Sire', 'Dam'])
    ped_wright = read_pedigree(wright_df)

    print(f"\nOrdered IDs: {ped_wright.get_ordered_ids_str()}")
    print(f"Inbreeding: {ped_wright.get_inbreeding_coefficients_ordered()}")
    # Expected F for P: F_P = 0.5 * A_DS
    # A_DS = 0.5*(A_DB + A_DC) = 0.5*( (0.5*(A_DX + A_D0)=0.5*A_DX) + (0.5*(A_SX + A_S0)=0.5*A_SX) )
    # Need to trace properly.
    # For P (child of D and S): F_P = 0.5 * A_DS
    # D and S are full sibs if B and C are same parents (X).
    # A_DS (full sibs) = 0.5 * (1 + F_X) (if X is common parent of B and C and not inbred)
    # If B and C are from X (founder, Fx=0), then B and C are half-sibs if other parents of B,C differ.
    # If B(X,0), C(X,0), then A_BC = 0.25 (assuming X not inbred).
    # S(B,C), D(B,C). So S and D are full sibs. F_S = F_D = 0.5 * A_BC.
    # If A_BC = 0.25, then F_S = F_D = 0.125.
    # A_DS (between full sibs S and D) = 0.5 * (1 + 0.5*(F_B + F_C))
    # F_B from (X,0) is 0. F_C from (X,0) is 0. So A_DS = 0.5 * (1 + 0) = 0.5.
    # Then F_P = 0.5 * A_DS = 0.5 * 0.5 = 0.25.
    # Check P's inbreeding:
    node_P = ped_wright.id_map.get('P')
    if node_P: print(f"Calculated F_P: {node_P.inbreeding_coeff}") # Should be 0.25

    A_inv_wright = calculate_A_inverse(ped_wright)
    print(f"\nA-Inverse for Wright's example ({A_inv_wright.shape}):")
    print(A_inv_wright.toarray())
```

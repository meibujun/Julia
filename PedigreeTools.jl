# Filename: PedigreeTools.jl

"""
The PedigreeTools module provides tools for processing, cleaning, validating, and sorting pedigree data for livestock breeding.
Key features include:
- Reading pedigree data from CSV files.
- Standardizing data formats (e.g., handling missing values).
- Validating data integrity: checking for duplicate IDs, supplementing missing ancestors.
- Detecting and reporting pedigree cycles (biologically impossible errors).
- Topologically sorting the pedigree to ensure parents appear before their offspring.
- Re-encoding individual IDs into consecutive integers for subsequent matrix operations, and providing ID mappings.
"""
module PedigreeTools

using DataFrames
using CSV
using Graphs

# Corrected export statement
export PedigreeRecord, Pedigree, clean_and_sort_pedigree, read_pedigree_from_csv, to_pedigree_struct

#=
--------------------------------------------------------------------------------
Part 1: Data Structure Definitions
--------------------------------------------------------------------------------
Define data structures for efficient storage and access of pedigree information.
=#

"""
    PedigreeRecord
A struct for storing a single individual's pedigree record.
"""
struct PedigreeRecord
    id::Int           # Individual ID (re-encoded)
    sire::Int         # Sire ID (re-encoded, 0 for unknown)
    dam::Int          # Dam ID (re-encoded, 0 for unknown)
end

"""
    Pedigree
A struct for storing the entire pedigree.
Contains a vector of records and a mapping from original ID to vector index for fast lookups.
"""
struct Pedigree
    records::Vector{PedigreeRecord}     # List of pedigree records
    orig_id_map::Dict{Any, Int}         # Mapping from original ID -> re-encoded ID (1 to N)
end


#=
--------------------------------------------------------------------------------
Part 2: Core Cleaning and Sorting Function
--------------------------------------------------------------------------------
This is the core of the module, responsible for executing the complete cleaning, validation, and sorting workflow.
=#

"""
    clean_and_sort_pedigree(df_in::DataFrame)

Performs comprehensive cleaning, validation, topological sorting, and ID re-encoding on the input pedigree DataFrame.

### Arguments:
- `df_in::DataFrame`: A DataFrame containing pedigree data, which should include columns for individual, sire, and dam IDs.
  Column names should be `:id`, `:sire`, `:dam` (or case variants).

### Returns:
A `NamedTuple` containing:
- `sorted_df::DataFrame`: The cleaned, sorted DataFrame with an added re-encoded ID column (`:recoded_id`).
- `orig_to_recoded::Dict`: A dictionary mapping original IDs to new encoded IDs.
- `recoded_to_orig::Dict`: A dictionary mapping new encoded IDs to original IDs.

### Cleaning and Validation Steps:
1.  **Standardization**: Unifies column names to lowercase and replaces `missing` values in parent ID columns with 0.
2.  **Check for Duplicate IDs**: Ensures that each individual's ID is unique.
3.  **Supplement Missing Ancestors**: Finds IDs that appear in the sire/dam columns but not in the id column and adds them as base individuals.
4.  **Topological Sort and Cycle Detection**: Uses a graph algorithm to topologically sort the pedigree and simultaneously detect any cycles.
5.  **ID Re-encoding**: Assigns a new consecutive ID from 1 to N to each individual based on the topological order.
"""
function clean_and_sort_pedigree(df_in::DataFrame)
    # 1. Standardization: Create a copy to avoid modifying the original data, and unify format
    df = copy(df_in)
    rename!(df, lowercase.(names(df))) # Unify column names to lowercase

    # Replace missing values in parent ID columns with 0
    df.sire = coalesce.(df.sire, 0)
    df.dam = coalesce.(df.dam, 0)

    println("Step 1: Data standardization complete.")

    # 2. Check for Duplicate IDs
    if nrow(unique(df, :id)) != nrow(df)
        dups = df[nonunique(df, :id), :id]
        error("Validation failed: Duplicate individual IDs found! Duplicate IDs are: $dups")
    end
    println("Step 2: ID uniqueness validation passed.")

    # 3. Supplement Missing Ancestors
    all_ids = Set(df.id)
    parent_ids = Set(vcat(df.sire, df.dam))
    delete!(parent_ids, 0) # Remove 0 from the set, as 0 is not a valid ID

    missing_ancestors = setdiff(parent_ids, all_ids) # Find the set difference

    if !isempty(missing_ancestors)
        println("Step 3: Found and supplemented $(length(missing_ancestors)) missing ancestors: $missing_ancestors")
        for anc in missing_ancestors
            push!(df, (id=anc, sire=0, dam=0))
        end
        # Update all_ids for subsequent use
        all_ids = Set(df.id)
    else
        println("Step 3: No missing ancestors, pedigree is complete.")
    end

    # 4. Topological Sort and Cycle Detection (using Graphs.jl)
    println("Step 4: Starting topological sort and cycle detection...")

    # Create a mapping from ID to graph vertex index
    # We use a sorted list of unique IDs to ensure the mapping is deterministic
    sorted_unique_ids = sort(collect(all_ids))
    id_to_vertex = Dict(id => i for (i, id) in enumerate(sorted_unique_ids))

    N = length(sorted_unique_ids)
    g = DiGraph(N)

    # Add edges: parent -> child
    for row in eachrow(df)
        if row.sire != 0
            add_edge!(g, id_to_vertex[row.sire], id_to_vertex[row.id])
        end
        if row.dam != 0
            add_edge!(g, id_to_vertex[row.dam], id_to_vertex[row.id])
        end
    end

    # Check for cycles
    if is_cyclic(g)
        error("Validation failed: A cycle exists in the pedigree, cannot perform topological sort. Please check the data source.")
    end

    # Get the topologically sorted vertex sequence
    topo_order_vertices = topological_sort(g)

    # Map vertex indices back to original IDs
    sorted_orig_ids = [sorted_unique_ids[i] for i in topo_order_vertices]
    println("Step 4: Topological sort complete.")

    # 5. ID Re-encoding and DataFrame Sorting
    println("Step 5: Starting ID re-encoding and final sorting...")

    # Create ID mapping dictionaries
    orig_to_recoded = Dict(orig_id => i for (i, orig_id) in enumerate(sorted_orig_ids))
    recoded_to_orig = Dict(i => orig_id for (i, orig_id) in enumerate(sorted_orig_ids))

    # Add new re-encoded ID columns
    df.recoded_id = [orig_to_recoded[x] for x in df.id]
    df.recoded_sire = [sire == 0 ? 0 : orig_to_recoded[sire] for sire in df.sire]
    df.recoded_dam = [dam == 0 ? 0 : orig_to_recoded[dam] for dam in df.dam]

    # Sort the DataFrame by the new re-encoded ID
    sort!(df, :recoded_id)
    println("Step 5: ID re-encoding complete.")

    return (
        sorted_df = df,
        orig_to_recoded = orig_to_recoded,
        recoded_to_orig = recoded_to_orig
    )
end


#=
--------------------------------------------------------------------------------
Part 3: Helper and Utility Functions
--------------------------------------------------------------------------------
=#

"""
    read_pedigree_from_csv(filepath::String; missingstring="NA")

Reads pedigree data from a CSV file and performs initial processing.

### Arguments:
- `filepath::String`: The path to the CSV file.
- `missingstring`: The string used in the file to represent unknown parents, defaults to "NA".

### Returns:
- A `DataFrame` that can be used with the `clean_and_sort_pedigree` function.
"""
function read_pedigree_from_csv(filepath::String; missingstring="NA")
    try
        df = CSV.read(filepath, DataFrame; missingstring=missingstring)
        println("Successfully read data from $filepath.")
        return df
    catch e
        error("Error reading file $filepath: $e")
    end
end


"""
    to_pedigree_struct(sorted_df::DataFrame, orig_to_recoded::Dict)

Converts a cleaned and sorted DataFrame into the more efficient `Pedigree` struct.
"""
function to_pedigree_struct(sorted_df::DataFrame, orig_to_recoded::Dict)
    records = PedigreeRecord[]
    for row in eachrow(sorted_df)
        push!(records, PedigreeRecord(row.recoded_id, row.recoded_sire, row.recoded_dam))
    end
    return Pedigree(records, orig_to_recoded)
end


end # module PedigreeTools


#=
--------------------------------------------------------------------------------
Part 4: Main Program Entry and Example
--------------------------------------------------------------------------------
This part is executed only when this file is run directly, to demonstrate the module's functionality.
=#

# Use the module we just defined
using .PedigreeTools

function main()
    println("=============== Pedigree Data Cleaning and Representation Demo ===============\\n")

    # 1. Create a sample CSV file with various cases to handle
    # - Unordered IDs
    # - "NA" for parent IDs
    # - A missing ancestor (ID 3)
    csv_data = """
    ID,Sire,Dam
    4,1,2
    5,1,2
    6,4,3
    7,4,5
    1,0,0
    2,NA,0
    """
    filepath = "pedigree_demo.csv"
    write(filepath, csv_data)
    println("Sample file created: $filepath\\n")

    # 2. Read data from the CSV file
    # Qualify function calls with the module name to avoid ambiguity
    df_raw = PedigreeTools.read_pedigree_from_csv(filepath)
    println("Original data:")
    println(df_raw, "\\n")

    # 3. Execute the core cleaning, sorting, and re-encoding workflow
    println("--- Starting cleaning, sorting, and re-encoding workflow ---\\n")
    result = PedigreeTools.clean_and_sort_pedigree(df_raw)
    println("\\n--- Workflow execution complete ---\\n")

    # 4. Display the results
    println("Final sorted and re-encoded DataFrame:")
    println(result.sorted_df, "\\n")

    println("Original ID -> New Encoded ID Mapping:")
    println(result.orig_to_recoded, "\\n")

    println("New Encoded ID -> Original ID Mapping:")
    println(result.recoded_to_orig, "\\n")

    # 5. Demonstrate converting the DataFrame to the custom Pedigree struct
    println("--- Converting to efficient Pedigree struct ---")
    ped_struct = PedigreeTools.to_pedigree_struct(result.sorted_df, result.orig_to_recoded)
    println("Pedigree struct created successfully.")
    println("Contains $(length(ped_struct.records)) records.")
    println("Sample records (first 3):")
    for i in 1:min(3, length(ped_struct.records))
        println(ped_struct.records[i])
    end
    println()

    # 6. Clean up the sample file
    rm(filepath)
    println("Sample file cleaned up.")

    println("=============== Demo End ===============\\n")
end

# When the file is executed directly, call the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

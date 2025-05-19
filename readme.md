# Relationship Coefficient Examples

This repository contains small examples for calculating relationship
coefficients used in animal breeding or pedigree analysis.

- `relationship_coefficient.jl` demonstrates the computation using Julia.
- `relationship_coefficient.py` provides an equivalent implementation in Python.

Both scripts build a relationship matrix from a simple pedigree and show the
coefficient between two half siblings `B` and `C`.

To run the Python version:

```bash
python3 relationship_coefficient.py
```

If you have Julia installed locally, you can run the Julia script with:

```bash
julia relationship_coefficient.jl
```

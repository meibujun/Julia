"""Generate and display the 9x9 multiplication table.

The script defines a function ``print_multiplication_table`` that prints
multiplication expressions in the classic triangular form used for the
"九九乘法表". Running this module will display the table to stdout.
"""

from __future__ import annotations


def print_multiplication_table(max_factor: int = 9) -> None:
    """Print a multiplication table up to ``max_factor``.

    Parameters
    ----------
    max_factor : int
        Highest multiplier in the table. Defaults to 9.
    """
    for i in range(1, max_factor + 1):
        line = []
        for j in range(1, i + 1):
            line.append(f"{j} × {i} = {i * j}")
        print("\t".join(line))


if __name__ == "__main__":
    print_multiplication_table()

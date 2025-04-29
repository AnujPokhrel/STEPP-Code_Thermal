#!/usr/bin/env python3
import os
import sys

def tree(path: str, depth: int, file_limit: int, prefix: str = "") -> None:
    """
    Recursively print a directory tree.

    Parameters
    ----------
    path        : starting directory
    depth       : how many levels below `path` to visit (0 == just `path`)
    file_limit  : max entries to show in *each* directory
    prefix      : indentation prefix (internal use only)
    """
    if depth < 0 or not os.path.isdir(path):
        return

    # Print the directory name itself (basename keeps output short)
    name = os.path.basename(os.path.abspath(path))
    print(f"{prefix}{name}/")

    # Nothing more to show?
    if depth == 0:
        return

    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        print(f"{prefix}└── [permission denied]")
        return

    shown = 0
    n = len(entries)
    for idx, entry in enumerate(entries):
        if shown >= file_limit:
            print(f"{prefix}└── ... (more items hidden)")
            break

        is_last = idx == n - 1 or shown == file_limit - 1
        connector = "└── " if is_last else "├── "
        new_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{prefix}{connector}{entry}")
        shown += 1

        sub_path = os.path.join(path, entry)
        if os.path.isdir(sub_path):
            tree(sub_path, depth - 1, file_limit, new_prefix)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <directory> <file_limit>")
        sys.exit(1)

    start_dir = sys.argv[1]
    limit = int(sys.argv[2])

    if not os.path.isdir(start_dir):
        print(f"Error: '{start_dir}' is not a directory.")
        sys.exit(1)

    # Show three levels below the start directory (total depth = 3)
    tree(start_dir, depth=3, file_limit=limit)

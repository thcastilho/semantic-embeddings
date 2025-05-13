# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (C) 2025 Thiago César Castilho Almeida et al.

def read_ranked_lists_file(file_path, top_k):
    """
    Reads a file containing ranked lists and returns the top-k entries for each line.

    Args:
        file_path (str): Path to the file containing ranked lists.
        top_k (int): Number of top entries to keep from each list.

    Returns:
        list[list[int]]: A list of top-k ranked lists.
    """
    print("\t\tReading file", file_path)
    with open(file_path, "r") as f:
        return [[int(y) for y in x.strip().split(" ")][:top_k] for x in f.readlines()]

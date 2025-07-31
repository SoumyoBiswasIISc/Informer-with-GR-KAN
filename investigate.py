#!/usr/bin/env python3
"""
Scan the Informer2020 repo at a fixed path and find occurrences of ArgumentParser variable names
in all files. Outputs a nested dictionary of the form:

{
    "<arg_name>": {
        "relative/path/to/file1.py": [line_num1, line_num2, ...],
        "relative/path/to/file2.txt": [line_num3, ...],
        ...
    },
    ...
}

Usage:
    Simply run:
        python find_informer_args.py
    (The script will use the hardcoded path for the Informer2020 repo.)
"""

import os
import re
import pprint

def extract_arg_names(main_file_path):
    """
    Parse main_informer.py to extract all argument names passed to parser.add_argument().
    Returns a list of variable names (strings) without leading dashes and with
    hyphens replaced by underscores.
    """
    arg_names = set()
    add_arg_pattern = re.compile(r"""add_argument\s*\(\s*['"](--[A-Za-z0-9\-_]+)['"]""")
    with open(main_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = add_arg_pattern.search(line)
            if match:
                long_opt = match.group(1)         # e.g. "--seq_len" or "--data-path"
                var_name = long_opt.lstrip('-')   # e.g. "seq_len" or "data-path"
                var_name = var_name.replace('-', '_')
                arg_names.add(var_name)
    return sorted(arg_names)

def scan_files_for_vars(repo_root, var_list):
    """
    Walk through all files under repo_root, searching each line for occurrences
    of each variable in var_list (using word-boundary matching). Returns a dict:
    {
        var1: { "fileA": [ln1, ln2, ...], "fileB": [ln3, ...], ... },
        var2: { ... },
        ...
    }
    """
    result = {var: {} for var in var_list}
    word_patterns = {var: re.compile(r'\b' + re.escape(var) + r'\b') for var in var_list}

    for dirpath, _, filenames in os.walk(repo_root):
        for fname in filenames:
            file_path = os.path.join(dirpath, fname)
            # Compute a relative path from repo_root for cleaner output
            rel_path = os.path.relpath(file_path, repo_root)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for lineno, line in enumerate(f, start=1):
                        for var, pattern in word_patterns.items():
                            if pattern.search(line):
                                result[var].setdefault(rel_path, []).append(lineno)
            except (UnicodeDecodeError, PermissionError):
                # Skip binary or unreadable files
                continue

    # Remove entries for vars that had no hits at all
    result = {var: files for var, files in result.items() if files}
    return result

def main():
    # Hardcoded path to the cloned Informer2020 repository:
    repo_root = "/user1/res/cvpr/soumyo.b_r/Understanding Informer/Informer2020"
    main_file = os.path.join(repo_root, 'main_informer.py')
    if not os.path.isfile(main_file):
        print(f"Error: could not find 'main_informer.py' at {main_file}")
        return

    # 1. Extract variable names from main_informer.py
    arg_vars = extract_arg_names(main_file)
    if not arg_vars:
        print("No parser.add_argument calls found in main_informer.py.")
        return

    # 2. Scan all files in the repo for these variable names
    occurrences = scan_files_for_vars(repo_root, arg_vars)

    # 3. Pretty-print the resulting dictionary
    pprint.pprint(occurrences)

if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
bin/parse_ii.py
===============

Flattened source files produced by compiler with "-E" option
contain the source and commented line markers indicating
the names of the headers with integer codes of form::

   # [line number] "[filename]" [flags]


Meanings of the flags:

* 1: Indicates the start of a new include file.
* 2: Indicates returning to a file after finishing an include file.
* 3: Indicates that the following text comes from a system header file,
     so certain warnings should be suppressed.
* 4: Indicates that the following text should be treated as wrapped
     in an implicit extern "C" block.

::

    [lo] A[blyth@localhost opticks]$ grep CSGOptiX7.cu /data1/blyth/local/opticks_Release_g411/build/CSGOptiX/CSGOptiX7_flattened.ii
    # 0 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 1 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 51 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 52 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 54 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 55 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 56 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 57 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 58 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 59 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 60 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 61 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 69 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 70 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 73 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 75 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 76 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 78 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 79 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 80 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 82 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 83 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 85 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 86 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2
    # 115 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 158 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 238 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 340 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 352 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 371 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 508 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 574 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 640 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 737 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 827 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    # 921 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
    [lo] A[blyth@localhost opticks]$


* Line 51 of CSGOptiX7.cu is first include (corresponds to line  36443 of flattened)
* Lines 115,158,...,921 correspond to lines with function definitions.


::

         1 # 0 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
     36442 # 1 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
     36443 # 51 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu"
     49287 # 52 "/home/blyth/opticks/CSGOptiX/CSGOptiX7.cu" 2






"""

import sys
from collections import defaultdict, Counter
import re

def parse_ii(filepath, target_query=None):
    file_lines = Counter()
    # Store child -> set of parents for reverse-lookup tracking
    parent_map = defaultdict(set)

    current_stack = [filepath]
    pp_pattern = re.compile(r'^\s*#\s*([0-9]+)\s+"([^"]+)"')

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = pp_pattern.match(line)
            if match:
                target_file = match.group(2)
                flags = line.split()[3:] if len(line.split()) > 3 else []

                if '1' in flags:
                    # enter a new include - associate parentage - collect path into current_stack
                    parent_map[target_file].add(current_stack[-1])
                    current_stack.append(target_file)
                elif '2' in flags:
                    # exit an include
                    if len(current_stack) > 1:
                        current_stack.pop()
                    pass
                else:
                    if target_file != current_stack[-1]:
                        if target_file in current_stack:
                            # pop stack to get back to target_file
                            while current_stack[-1] != target_file and len(current_stack) > 1:
                                current_stack.pop()
                            pass
                        else:
                            parent_map[target_file].add(current_stack[-1])
                            current_stack.append(target_file)
                        pass
                    pass
                pass
            else:
                if current_stack:
                    file_lines[current_stack[-1]] += 1
                pass
            pass
        pass
    pass

    if target_query:
        print(f"\n[Ancestry Trace for files matching: '{target_query}']=----------------")
        matches = [f for f in parent_map.keys() if target_query in f]
        for m in matches:
            print(f"\nTarget: {m}")
            print("Included by (Parents):")
            # Trace backwards up parents
            print_chain(m, parent_map, depth=0, max_depth=6)
        pass
        print("-" * 90)
    else:
        # Standard summary table
        sorted_files = file_lines.most_common()
        print(f"{'Lines':<10} {'Header / Source File'}")
        print("-" * 90)
        for fname, count in sorted_files:
            print(f"{count:<10} {fname}")
        pass
    pass

def print_chain(node, parent_map, depth, max_depth):
    if depth > max_depth or node not in parent_map:
        return
    for p in sorted(parent_map[node]):
        print("  " * depth + f"  ^--- {p}")
        print_chain(p, parent_map, depth + 1, max_depth)
    pass

if __name__ == '__main__':
    if len(sys.argv) > 1:
        ii_file = sys.argv[1]
        # Optional second argument allows querying who includes a specific file/keyword (e.g., 'string')
        query = sys.argv[2] if len(sys.argv) > 2 else None
        parse_ii(ii_file, query)
    else:
        print(f"Usage: {sys.argv[0]} <path_to_flattened.ii> [search_query]")
    pass


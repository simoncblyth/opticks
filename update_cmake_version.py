#!/usr/bin/env python3
"""
update_cmake_version.py
=======================

The first line of package CMakeLists.txt is typically of form::

    cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

Typical usage::

    cd ~/opticks
    vi ./update_cmake_version.py ## increase CMAKE_MINIMUM_VERSION in the below with a note describing why
    ./update_cmake_version.py    ## update the top level package CMakeLists.txt


"""
import argparse
import re
from pathlib import Path

def update_cmake_version(root_path: Path, target_version: str, every: bool):
    root_path = root_path.resolve()

    if every:
        # Recursively find all CMakeLists.txt files
        cmake_files = list(root_path.rglob('CMakeLists.txt'))
    else:
        # Target direct subdirectories containing a CMakeLists.txt
        cmake_files = []
        for item in root_path.iterdir():
            if item.is_dir():
                pkg_cmake = item / 'CMakeLists.txt'
                if pkg_cmake.is_file():
                    cmake_files.append(pkg_cmake)
                pass
            pass
        pass
    pass
    # Regex to match cmake_minimum_required(VERSION x.xx ...) case-insensitively
    pattern = re.compile(r'(cmake_minimum_required\s*\(\s*VERSION\s+)[\d\.]+', re.IGNORECASE)

    updated_count = 0
    for file_path in cmake_files:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            new_content, count = pattern.subn(rf'\g<1>{target_version}', content)

            if count > 0:
                if content != new_content:
                    file_path.write_text(new_content, encoding='utf-8')
                    print(f"[UPDATED] {file_path.relative_to(root_path)} -> VERSION {target_version}")
                    updated_count += 1
                else:
                    print(f"[OK]      {file_path.relative_to(root_path)} is already at version {target_version}")
            else:
                print(f"[WARNING] No cmake_minimum_required found in {file_path.relative_to(root_path)}")
            pass
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    print(f"\nSuccessfully updated {updated_count} file(s).")

if __name__ == '__main__':

    CMAKE_MINIMUM_VERSION = '3.27'   # bumped for optixir CUDA_OPTIX_COMPILATION support
    parser = argparse.ArgumentParser(description="Manage cmake_minimum_required versions across Opticks packages.")
    parser.add_argument('--root', type=str, default='.', help='Path to the Opticks repository root (default: current dir).')
    parser.add_argument('--version', type=str, default=CMAKE_MINIMUM_VERSION, help=f'Target minimum CMake version (e.g., {CMAKE_MINIMUM_VERSION}.')
    parser.add_argument('--every', action='store_true', help='Update every CMakeLists.txt not just the typical top-level package CMakeLists.txt files.')

    args = parser.parse_args()
    update_cmake_version(Path(args.root), args.version, args.every)
    pass

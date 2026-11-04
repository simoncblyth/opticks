#!/usr/bin/env bash

# Check if exactly 3 arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <file_path> <first_line> <last_line>" >&2
    exit 1
fi

FILE="$1"
START="$2"
END="$3"

# Verify that the file exists and is readable
if [ ! -f "$FILE" ]; then
    echo "Error: File '$FILE' not found or is not a regular file." >&2
    exit 1
fi

# Use sed to extract the line range
# -n suppresses automatic printing, and ${START},${END}p prints only the specified range
sed -n "${START},${END}p" "$FILE"

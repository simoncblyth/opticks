#!/usr/bin/env python
"""
OBSOLETE.py
===========

Place marker OBSOLETE.txt files within obsolete packages that 
are not yet willing to delete

"""

from pathlib import Path

OBSOLETE_PKG = [
   "cfg4", 
   "okop",
   "thrustrap",
   "numpyserver",
   "extg4",
   "c4",
   "g4ok",
   "okg4",
   "optixrap",
   "opticksgl",
   "cudarap",
   "npy",
   "oglrap",
   "ok"
]

if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent
    print(f"OBSOLETE.py - SCRIPT_DIR {SCRIPT_DIR} ROOT_DIR {ROOT_DIR}")
    for pkg in ROOT_DIR.iterdir():
        if not pkg.is_dir(): continue
        if pkg.name in OBSOLETE_PKG:
            MARKER = pkg / "OBSOLETE.txt"
            if not MARKER.is_file():
                print(f"Writing MARKER : {MARKER}")
                MARKER.write_text(f"THIS {pkg.name} PACKAGE IS OBSOLETE - AND WILL AT SOME POINT BE REMOVED - SEE bin/OBSOLETE.py\n")
            else:
                print(f"MARKER file exists already : {MARKER}")
            pass 
        pass
    pass
pass




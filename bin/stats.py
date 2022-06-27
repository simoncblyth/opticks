#!/usr/bin/env python
"""
stats.py 
==========

Count the number of files with different entensions in each package directory 
and create an RST table presenting this. 

::

    epsilon:opticks blyth$ ./bin/stats.py 
    Pkg : 350 : /Users/blyth/opticks/ana : {'.py': 239, '.pyc': 47, '.sh': 31, '.rst': 20, '.cc': 3, '.bash': 3, '.txt': 1, '.hh': 1, '': 4, '.swp': 1} 
    Pkg :  34 : /Users/blyth/opticks/okop : {'.hh': 14, '.cc': 10, '.sh': 3, '.txt': 2, '.old': 1, '': 1, '.rst': 1, '.bash': 1, '.cu': 1} 
    Pkg :  96 : /Users/blyth/opticks/CSGOptiX : {'.cc': 14, '.sh': 50, '.h': 18, '.rst': 1, '.txt': 1, '': 2, '.bash': 1, '.py': 2, '.cu': 4, '.hh': 2, '.swp': 1} 
    Pkg :  32 : /Users/blyth/opticks/cudarap : {'.hh': 14, '.cc': 8, '.txt': 1, '.cu': 3, '.sh': 1, '.old': 1, '': 1, '.h': 2, '.bash': 1} 
    Pkg :  87 : /Users/blyth/opticks/CSG : {'.h': 35, '.sh': 17, '.cc': 18, '.rst': 1, '.txt': 1, '.py': 7, '': 3, '.pyc': 2, '.hh': 2, '.in': 1} 
    Pkg :  23 : /Users/blyth/opticks/opticksgeo : {'.hh': 10, '.cc': 6, '.rst': 1, '.sh': 1, '.txt': 1, '.bash': 1, '.old': 1, '': 2} 
    Pkg : 262 : /Users/blyth/opticks/cfg4 : {'.cc': 117, '.h': 9, '.hh': 117, '.py': 3, '.rst': 7, '.sh': 4, '.txt': 2, '.bash': 1, '': 1, '.old': 1}

"""
import os, numpy as np
from opticks.ana.rsttable import RSTTable


class Pkg(object):
    EXCLUDE = ".pyc .log .swp .txt .rst .in .old .sh .bash .cfg".split()
    EXTS = ".hh .h .hpp .cc .cpp .cu .py".split()
    def __init__(self, fold):
        names = os.listdir(fold)
        pkg = os.path.basename(fold)

        exts = {}
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext == "" or ext in self.EXCLUDE: continue
            if not ext in self.EXTS: print("unexpected ext %s " % ext )
            if not ext in exts: exts[ext] = 0
            exts[ext]+=1   
        pass
        stats = np.zeros( (1+len(self.EXTS),), dtype=np.object )

        stats[0] = pkg
        for i, ext in enumerate(self.EXTS): 
            stats[1+i] = exts.get(ext, 0)
        pass
        self.fold = fold
        self.pkg = pkg
        self.names = names
        self.exts = exts  
        self.stats = stats 

    def __repr__(self):
        return "Pkg : %3d : %15s : %s " % (len(self.names), self.pkg, repr(self.exts)) 
 

class Stats(object):
    HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def __init__(self, home=HOME):
        pkgs = [] 
        for dirpath, dirs, names in os.walk(home):
            if "CMakeLists.txt" in names:
                if dirpath == home or dirpath.find("examples") > -1 or dirpath.find("tests") > 1: continue
                pkg = Pkg(dirpath) 
                pkgs.append(pkg)
            pass 
        pass
        stats = np.zeros( ( len(pkgs), 1+len(Pkg.EXTS) ), dtype=np.object )
        for i, pkg in enumerate(pkgs):
            stats[i] = pkg.stats
        pass

        self.pkgs = pkgs
        self.stats = stats
    pass
    def __str__(self):
        labels = ["pkg"]+Pkg.EXTS ;   
        return RSTTable.Rdr(self.stats, labels, rfm="%10d", left_wid=15, left_rfm="%15s", left_hfm="%15s" )
    def __repr__(self):
        return "\n".join(list(map(repr, self.pkgs)))
    pass

if __name__ == '__main__':
    st = Stats()
    print(st)
     


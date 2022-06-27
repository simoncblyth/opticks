#!/usr/bin/env python
"""
stats.py 
==========

Count the number of files with different entensions in each package directory 
and create an RST table presenting this.::

    cd ~/opticks/bin
    ./stats.sh 

::

    epsilon:bin blyth$ ./stats.sh 
    +---------------+----------+----------+----------+----------+----------+----------+----------+
    |            pkg|       .hh|        .h|      .hpp|       .cc|      .cpp|       .cu|       .py|
    +===============+==========+==========+==========+==========+==========+==========+==========+
    |            ana|         1|         0|         0|         3|         0|         0|       239|
    +---------------+----------+----------+----------+----------+----------+----------+----------+
    |           okop|        14|         0|         0|        10|         0|         1|         0|
    +---------------+----------+----------+----------+----------+----------+----------+----------+
    |       CSGOptiX|         2|        18|         0|        14|         0|         4|         2|
    +---------------+----------+----------+----------+----------+----------+----------+----------+
    |        cudarap|        14|         2|         0|         8|         0|         3|         0|
    +---------------+----------+----------+----------+----------+----------+----------+----------+

TODO:

0. dependency ordering of packages (see bin/CMakeLists.py)
1. presentation selection, pkg notes in right hand column  
2. consolidate .hh/.hpp and .cc/.cpp

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
     


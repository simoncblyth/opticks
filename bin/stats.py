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
import os 


class Pkg(object):
    def __init__(self, fold):
        names = os.listdir(fold)

        exts = {}
        for name in names:
            stem, ext = os.path.splitext(name)
            if not ext in exts: exts[ext] = 0
            exts[ext]+=1   
        pass
        self.fold = fold
        self.names = names
        self.exts = exts  


    def __repr__(self):
        return "Pkg : %3d : %s : %s " % (len(self.names), self.fold, repr(self.exts)) 
 

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
        self.pkgs = pkgs
    pass
    def __repr__(self):
        return "\n".join(list(map(repr, self.pkgs)))
    pass

if __name__ == '__main__':
    st = Stats()
    print(st)
     


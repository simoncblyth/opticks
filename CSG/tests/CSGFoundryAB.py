#!/usr/bin/env python 

from opticks.CSG.CSGFoundry import CSGFoundry 

if __name__ == '__main__':

    a = CSGFoundry.Load("$A_CFBASE", symbol="a")
    b = CSGFoundry.Load("$B_CFBASE", symbol="b")

    print(a.brief())
    print(b.brief())



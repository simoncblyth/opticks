#!/usr/bin/env python 


from opticks.ana.fold import Fold
from opticks.CSG.CSGFoundry import CSGFoundry 

if __name__ == '__main__':


    A = Fold.Load("$A_CFBASE/CSGFoundry", symbol="A")
    B = Fold.Load("$B_CFBASE/CSGFoundry", symbol="B")

    print(repr(A))
    print(repr(B))

if 0:

    a = CSGFoundry.Load("$A_CFBASE", symbol="a")
    b = CSGFoundry.Load("$B_CFBASE", symbol="b")
    print(a.brief())
    print(b.brief())



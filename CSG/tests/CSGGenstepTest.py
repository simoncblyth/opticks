#!/usr/bin/env python

from opticks.ana.fold import Fold

if __name__ == '__main__':
    fold = Fold.Load("$TMP/CSG/CSGGenstepTest/$MOI")
    print(fold)

    gs = fold.gs
    pp = fold.pp





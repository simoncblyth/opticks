#!/usr/bin/env python

from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load("/tmp/QPMTTest", symbol="t")
    #t = Fold.Load("/tmp/QPMTTest/rindex", symbol="t")
    print(repr(t))



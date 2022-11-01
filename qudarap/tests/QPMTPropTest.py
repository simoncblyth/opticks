#!/usr/bin/env python

from opticks.ana.fold import Fold

if __name__ == '__main__':
    t = Fold.Load("/tmp/QPMTPropTest/rindex", symbol="t")
    print(repr(t))



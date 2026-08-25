#!/usr/bin/env python

from opticks.ana.fold import Fold

if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    print(f.MaterialConstProperty_names)
    print(f.MaterialConstProperty)



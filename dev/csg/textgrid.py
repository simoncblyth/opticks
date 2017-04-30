#!/usr/bin/env python
"""
::

    In [36]: run textgrid.py
     E0  C1  C2  C3  C4  C5  C6  C7  C8  F0

     E1  A1   .   .   .   .   .   .  B8  F1

     E2   .  A2   .   .   .   .  B7   .  F2

     E3   .   .  A3   .   .  B6   .   .  F3

     E4   .   .   .  A4  B5   .   .   .  F4

     E5   .   .   .  B4  A5   .   .   .  F5

     E6   .   .  B3   .   .  A6   .   .  F6

     E7   .  B2   .   .   .   .  A7   .  F7

     E8  B1   .   .   .   .   .   .  A8  F8

     E9  D1  D2  D3  D4  D5  D6  D7  D8  F9


"""
import logging
log = logging.getLogger(__name__)
import numpy as np

class T(np.ndarray):
    @classmethod
    def init(cls, a, itemfmt="%3s", rowjoin="\n", empty=""):  
        assert len(a.shape) == 2, a
        t = a.view(cls)
        t.itemfmt = itemfmt
        t.rowjoin = rowjoin
        t.empty = empty 
        return t

    def __repr__(self):
        row_ = lambda r:" ".join(map(lambda _:self.itemfmt % (_ if _ is not None else self.empty) ,r))
        tab_ = lambda a:self.rowjoin.join(map(row_, a))
        return tab_(self)


class TextGrid(object):
    def __init__(self, ni, nj, **kwa):
        """
        grid of None (might produce gibberish in some imps?)
        dont want to use zeros as then get zeros at every spot on the grid
        """
        a = np.empty((ni,nj),dtype=np.object)  
        t = T.init(a, **kwa)
        self.a = a
        self.t = t 

    def __str__(self):
        return repr(self.t)



if __name__ == '__main__':
    pass

    n = 10 
    tg = TextGrid(n,n, itemfmt="%3s", rowjoin="\n\n", empty=".") 

    for k in range(n): 
        tg.a[k,k] = "A%d"%k       # diagonal from top-left to bottom-right
        tg.a[n-1-k,k] = "B%d"%k   # diagonal from bottom-left to top right
        tg.a[0,k] = "C%d"%k       # top row
        tg.a[-1,k] = "D%d"%k      # bottom row
        tg.a[k,0] = "E%d"%k       # left column 
        tg.a[k,-1] = "F%d"%k      # right column
    pass


    print tg 














     


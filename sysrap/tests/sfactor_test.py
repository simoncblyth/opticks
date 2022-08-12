#!/usr/bin/env python
"""
sfactor_test.py
================

::

    In [4]: f.factors[:,2:].copy().view("|S32")
    Out[4]: 
    array([[b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef'],
           [b'0123456789abcdef0123456789abcdef']], dtype='|S32')

"""
import numpy as np
from opticks.ana.fold import Fold

if __name__ == '__main__':
     f = Fold.Load(symbol="f") 
     print(repr(f))



#!/usr/bin/env python
"""
sseq_index_test.py : just for the NumPy statistical methods
==============================================================

Need this python despite most of comparison done in C++
as getting chi2 cdf in C++ is too involved


1. load persisted c2 info from $FOLD
2. do statistical null-hyp calc and report

null-hyp consistent means there is no significant difference between
the frequency counts in the A and B samples at a certain confidence
level (normally 5%)

"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.ana.nbase import chi2_pvalue


if __name__ == '__main__':
    print("[sseq_index_test.py\n")

    f = Fold.Load(symbol="f")  ## default arg is $FOLD
    print(repr(f))

    c2 = f.sseq_index_ab_chi2
    print(c2)

    c2sum = c2[0]
    c2n   = c2[1]
    c2cut = c2[2]

    c2per = c2sum/c2n

    c2pv = chi2_pvalue( c2sum, int(c2n) )
    c2pvm = "> 0.05 : null-hyp " if c2pv > 0.05 else "< 0.05 : NOT:null-hyp "
    c2pvd = "pv[%4.3f,%s] " % (c2pv, c2pvm)

    c2desc = "c2sum/c2n:c2per(C2CUT)  %5.2f/%d:%5.3f (%2d) %s" % ( c2sum, int(c2n), c2per, c2cut, c2pvd )
    c2label = "c2sum : %10.4f c2n : %10.4f c2per: %10.4f  C2CUT: %4d " % ( c2sum, c2n, c2per, c2cut )

    print(c2desc)
    print(c2label)
    print("]sseq_index_test.py\n")
pass

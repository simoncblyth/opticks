#!/usr/bin/env python
"""
tests/sphoton_test.py
======================

~/o/sysrap/tests/sphoton_test.sh


"""

import numpy as np, os, sys
from opticks.ana.fold import Fold
from opticks.sysrap.sphoton import SPhoton

import matplotlib.pyplot as plt
SIZE = np.array([1280,720])



def test_dot_pol_cross_mom_nrm(a):
    title = "test_dot_pol_cross_mom_nrm"
    fig, ax = plt.subplots(1, figsize=SIZE/100.)
    fig.suptitle(title)

    ax.scatter(  a[:,0], a[:,1], label="a[:,0] [:,1]  fr, pot " )
    ax.scatter(  a[:,0], a[:,2], label="a[:,0] a[:,2] fr, pot/mct " )
    ax.scatter(  a[:,0], a[:,3], label="a[:,0] a[:,3] fr, pot/st " )

    ax.plot( [0,1], [-1,-1],  label="-1" )
    ax.plot( [0,1], [+1,+1],  label="+1" )

    ax.legend()
    fig.show()


def test_make_record_array(a):
    title = "test_make_record_array"

    fig, ax = plt.subplots(1, figsize=SIZE/100.)
    fig.suptitle(title)
    ax.set_aspect(1)

    X,Y,Z = 0,1,2
    H,V = X,Z

    for j in range(10):
        ax.scatter( a[:,j,0,H], a[:,j,0,V], label=str(j) )
    pass
    ax.legend()
    fig.show()


def test_demoarray(a):
    print("[test_demoarray repr(a): ", repr(a))
    print("]test_demoarray")


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    TEST = os.environ.get("TEST", "demoarray")
    a = getattr(t,TEST, None)
    if a is None: sys.exit(0)

    if TEST == "dot_pol_cross_mom_nrm":
         test_dot_pol_cross_mom_nrm(a)
    elif TEST == "record":
        test_make_record_array(a)
    elif TEST == "demoarray":
        #test_demoarray(a)
        p = SPhoton.view(a)
    else:
        print("TEST:%s unhandled " % TEST)
    pass
pass



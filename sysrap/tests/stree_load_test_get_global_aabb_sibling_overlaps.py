#!/usr/bin/env python
"""
stree_load_test_get_global_aabb_sibling_overlaps.py
====================================================

::

    TEST=get_global_aabb_sibling_overlaps ~/o/sysrap/tests/stree_load_test.sh pdb


aabb overlap does not mean actual volumes overlap but it does
give candidates to check further for actual overlaps.

* TODO: make connection between snode indices "nidx"
  and the pointcloud identifiers from cx/cxt_min.py

  * see stree::populate_prim_nidx


"""
import os
import numpy as np
from opticks.ana.fold import Fold

np.set_printoptions(edgeitems=16)


if __name__ == '__main__':
    f = Fold.Load("$TMPFOLD/$TEST", symbol="f")
    print(repr(f))

    soname = np.array(list(filter(None,open(os.path.expandvars("$FOLD/soname_names.txt"),"r").read().split("\n"))))
    prname = np.array(list(filter(None,open(os.path.expandvars("$FOLD/prname_names.txt"),"r").read().split("\n"))))

    m2w = np.load(os.path.expandvars("$FOLD/m2w.npy")) # allows lookup of globalPrimIdx from nidx
    nidx_prim = np.load(os.path.expandvars("$FOLD/nidx_prim.npy")) # allows lookup of globalPrimIdx from nidx
    prim_nidx = np.load(os.path.expandvars("$FOLD/prim_nidx.npy")) # allows lookup of first nidx from globalPrimIdx


    bb = f.bb.reshape(-1,2,6)
    nn = f.nn.reshape(-1,2,15)

    print(f" bb {bb.shape}")
    print(f" nn {nn.shape}")


    lvn = "s_EMFsupport_ring21"  ## PICK LVN WITH ONLY ONE VOLUME FOR CLARITY
    lv = np.where( soname == lvn )[0][0]  # lookup lv from the name
    assert( soname[lv] == lvn )


    print(f" lvn {lvn} lv {lv}  soname[lv] {soname[lv]} ")

    w0 = np.where( nn[:,0,7] == lv )
    w1 = np.where( nn[:,1,7] == lv )

    print(f" w0 {w0}  select aabb overlap snode pairs where the first has the lv ")
    print(f" w1 {w1}  select aabb overlap snode pairs where the second has the lv ")


    n0 = soname[nn[w0][:,1,7]]   # names on one side
    n1 = soname[nn[w1][:,0,7]]   # names on other side

    print(f" n0 {n0} ")
    print(f" n1 {n1} ")


    i0_ = nn[w0][:,0,0]
    assert np.all( i0_ == i0_[0] )   ## all same as are checking overlaps of first node with others
    i0 = i0_[0]
    gpi0 = nidx_prim[i0]
    gpi0_prname = prname[gpi0]
    print(f" i0 {i0} gpi0 {gpi0} gpi0_prname {gpi0_prname} ")


    i1_ = nn[w1][:,0,0]           ## often multiple different nidx that are bbox overlapping
    gpi1 = nidx_prim[i1_]         ## THIS gpi1 CAN PROVIDE CONNECTION WITH simtrace point clouds
    gpi1_prname = prname[gpi1]

    print(f"i1_\n{i1_}\ngpi1\n{gpi1}\ngpi1_prname\n{gpi1_prname}\n")
pass


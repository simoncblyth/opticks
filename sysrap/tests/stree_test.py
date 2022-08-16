#!/usr/bin/env python

import numpy as np
from numpy.linalg import multi_dot

from opticks.ana.fold import Fold
from opticks.CSG.CSGFoundry import CSGFoundry 
from opticks.ana.eprint import eprint, epr
from opticks.sysrap.stree import stree


def check_inverse_pair(f, a_="inst", b_="iinst" ):
    """
    :param f: Fold instance

    * Now avoided clearing the identity by taking a copy 
    * This demonstrates the necessity for double precision transforms

    """
    a = getattr(f, a_).copy()
    b = getattr(f, b_).copy()

    a[:,:,3] = [0.,0.,0.,1.]
    b[:,:,3] = [0.,0.,0.,1.]

    assert len(a) == len(b)

    chk = np.zeros( (len(a), 4, 4) )   
    for i in range(len(a)): chk[i] = np.dot( a[i], b[i] )   
    chk -= np.eye(4)

    print("check_inverse_pair :  %s %s " % (a_, b_))
    print(chk.min(), chk.max())


def compare_f_with_cf(f, cf ):
    print("\n[compare_f_with_cf")
    eprint("(cf.inst[:,:,:3] - f.inst_f4[:,:,:3]).max()", globals(), locals())  
    eprint("(cf.inst[:,:,:3] - f.inst_f4[:,:,:3]).min()", globals(), locals())  
    eprint("np.all( cf.inst[:,:,3].view(np.int32)  == f.inst_f4[:,:,3].view(np.int32))", globals(), locals()) 
    print("]compare_f_with_cf\n")


def check_inst(f, cf):
    print("\n[check_inst(f,cf) : comparing f:stree inst(from U4Tree) with cf:CSGFoundry inst (via CSG_GGeo) ")

    eprint("np.abs(cf.inst[:,:,:3]-f.inst_f4[:,:,:3]).max()", globals(), locals() ) 
    w = epr("w = np.where( np.abs(cf.inst[:,:,:3]-f.inst_f4[:,:,:3]) > 0.0001 )",  globals(), locals() )

    check_inverse_pair(f, "inst", "iinst" )
    check_inverse_pair(f, "inst_f4", "iinst_f4" )
    compare_f_with_cf(f, cf ) 

    a_inst = cf.inst.copy() 
    b_inst = f.inst_f4.copy() 

    print("a_inst[-1]")
    print(a_inst[-1])
    print("b_inst[-1]")
    print(b_inst[-1])

    print("\n]check_inst(f,cf)")

def check_sensor(st):
    """
    dsid = np.diff(sid) 
    np.where( dsid != 1 )    (array([17611, 43211]),)

    In [40]: sid[17607:17618]
    Out[40]: array([ 17607,  17608,  17609,  17610,  17611, 300000, 300001, 300002, 300003, 300004, 300005], dtype=int32)

    In [42]: sid[43205:43220]
    Out[42]: array([325593, 325594, 325595, 325596, 325597, 325598, 325599,  30000,  30001,  30002,  30003,  30004,  30005,  30006,  30007], dtype=int32)

    """
    ws = np.where( st.nds.sensor_id > -1 )[0]  

    sidx_ = st.nds.sensor_index[ws]   
    sidx = sidx_[np.argsort(sidx_)]  
    x_sidx = np.arange(len(sidx), dtype=np.int32) 
    assert np.all( sidx == x_sidx ), "check sensor_idx contiguous from 0"   

    sid = st.nds.sensor_id[ws]  
    usid = np.unique(sid)
    assert len(usid) == len(sid), "check sensor_id are all unique"
    



if __name__ == '__main__':

    cf = CSGFoundry.Load()
    print(cf)

    f = Fold.Load(symbol="f")
    print(repr(f))

    g = Fold.Load(os.path.join(os.path.dirname(f.base), "GGeo/stree"), symbol="g") 

    st = stree(f)
    print(repr(st))

    check_inst(f, cf)
    #check_sensor(st)





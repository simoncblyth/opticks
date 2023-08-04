#!/usr/bin/env python
"""
gx.py : examine geometry from G4CXOpticks::saveGeometry 
==========================================================

"""

import numpy as np
from opticks.ana.fold import Fold
from opticks.sysrap.stree import stree
from opticks.CSG.CSGFoundry import CSGFoundry


def desc_sensor(st):
    """
    desc_sensor
        nds :  lv : soname
       4997 : 106 : HamamatsuR12860_PMT_20inch_inner1_solid_I 
       4997 : 108 : HamamatsuR12860_PMT_20inch_body_solid_1_4 
      12615 : 113 : NNVTMCPPMT_PMT_20inch_inner1_solid_head 
      12615 : 115 : NNVTMCPPMT_PMT_20inch_body_solid_head 
      25600 : 118 : PMT_3inch_inner1_solid_ell_helper 
      25600 : 120 : PMT_3inch_body_solid_ell_ell_helper 
       2400 : 130 : PMT_20inch_veto_inner1_solid 
       2400 : 132 : PMT_20inch_veto_body_solid_1_2 


    In [7]: np.unique( st.nds.sensor_id   )
    Out[7]: array([    -1,      0,      1,      2,      3, ..., 325595, 325596, 325597, 325598, 325599], dtype=int32)

    In [8]: np.unique( st.nds.sensor_index   )
    Out[8]: array([   -1,     0,     1,     2,     3, ..., 45607, 45608, 45609, 45610, 45611], dtype=int32)

    In [9]: np.unique( st.nds.sensor_name   )
    Out[9]: array([-1,  0,  1,  2,  3], dtype=int32)

    """
    ws = np.where(st.nds.sensor_index > -1 )[0]
    se = st.nds.sensor_index[ws]
    xse = np.arange(len(se), dtype=np.int32)    
    assert np.all( xse == se )  
    ulv, nlv = np.unique(st.nds.lvid[ws], return_counts=True)

    hfmt = "%7s : %3s : %50s : %s "
    fmt = "%7d : %3d : %50s : %s " 
    hdr = hfmt % ("nds", "lv", "soname", "0th" )

    zths = [st.find_lvid_node(ulv[i],0) for i in range(len(ulv))]

    extra = [] 
    for zth in zths:
        extra += ["zth:%s" % zth,]
        extra += [st.desc_nodes( st.get_children(zth, prepend_arg=True),brief=True),]
    pass
   
    head = ["desc_sensor",hdr]
    body = [fmt % ( nlv[i], ulv[i], st.soname_[ulv[i]], zths[i] ) for i in range(len(ulv))]
    tail = [hfmt % ( nlv.sum(), "", "", "" ),]
    return "\n".join(head+body+tail+extra)


if __name__ == '__main__':
    cf = CSGFoundry.Load(symbol="cf")
    print(cf) 

    f = Fold.Load(cf.base, "CSGFoundry/SSim/stree", symbol="f")
    st = stree(f)

    print(repr(st))
    print(desc_sensor(st))
pass
        


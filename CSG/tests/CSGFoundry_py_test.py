#!/usr/bin/env python
"""
CSGFoundry_py_test.py
========================

::
 
   ct ; ./CSGFoundry_py_test.sh


"""
import os, textwrap, numpy as np, logging
log = logging.getLogger(__name__)
from opticks.CSG.CSGFoundry import CSGFoundry 

if __name__ == '__main__':
    cf = CSGFoundry.Load()
    print(repr(cf))

    ins = cf.inst[:,0,3].view(np.int32)  # instance_idx      
    gas = cf.inst[:,1,3].view(np.int32)  # gas_idx      
    sid = cf.inst[:,2,3].view(np.int32)  # sensor_id       
    six = cf.inst[:,3,3].view(np.int32)  # sensor_idx       

    ugas,ngas = np.unique(gas, return_counts=True)

    EXPR = list(filter(None,textwrap.dedent(r"""
    cf.inst[:,:,3].view(np.int32)
    (sid.min(), sid.max())
    (six.min(), six.max())
    np.c_[ugas,ngas,cf.mmlabel] 

    np.c_[np.unique(sid[gas==0],return_counts=True)]     
    np.c_[np.unique(sid[gas==1],return_counts=True)]     
    np.c_[np.unique(sid[gas==2],return_counts=True)]     
    np.c_[np.unique(sid[gas==3],return_counts=True)]     
    np.c_[np.unique(sid[gas==4],return_counts=True)]     
    np.c_[np.unique(sid[gas==5],return_counts=True)]     
    np.c_[np.unique(sid[gas==6],return_counts=True)]     
    np.c_[np.unique(sid[gas==7],return_counts=True)]     
    np.c_[np.unique(sid[gas==8],return_counts=True)]     
    np.c_[np.unique(sid[gas==9],return_counts=True)]     


    """).split("\n")))

    for expr in EXPR:
        print(expr)
        if expr[0] == ' ': continue 
        print(eval(expr))
    pass

    sid2 = cf.sim.stree.sensor_id[:,0]  

     


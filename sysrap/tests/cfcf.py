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
     
    CAUTION : this clears the identity info prior to checking transforms
    """
    a = getattr(f, a_)
    b = getattr(f, b_)

    a[:,:,3] = [0.,0.,0.,1.]
    b[:,:,3] = [0.,0.,0.,1.]

    assert len(a) == len(b)

    chk = np.zeros( (len(a), 4, 4) )   
    for i in range(len(a)): chk[i] = np.dot( a[i], b[i] )   
    chk -= np.eye(4)

    print("check_inverse_pair :  %s %s " % (a_, b_))
    print(chk.min(), chk.max())


def compare_f_with_cf(f, cf ):
    print("compare_f_with_cf")
    eprint("(cf.inst - f.inst_f4).max()", globals(), locals())  
    eprint("(cf.inst - f.inst_f4).min()", globals(), locals())  


def check_inst(f, cf):

    eprint("np.abs(cf.inst-f.inst_f4).max()", globals(), locals() ) 
    w = epr("w = np.where( np.abs(cf.inst-f.inst_f4) > 0.0001 )",  globals(), locals() )

    check_inverse_pair(f, "inst", "iinst" )
    check_inverse_pair(f, "inst_f4", "iinst_f4" )
    compare_f_with_cf(f, cf ) 

    a_inst = cf.inst.copy() 
    b_inst = f.inst_f4.copy() 

    print("a_inst[-1]")
    print(a_inst[-1])
    print("b_inst[-1]")
    print(b_inst[-1])



if __name__ == '__main__':

    acf = CSGFoundry.Load("$A_CFBASE")
    print(acf)

    bcf = CSGFoundry.Load("$B_CFBASE")
    print(bcf)

    """

        node :        (23518, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/node.npy 
         itra :         (8159, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/itra.npy 
     meshname :               (139,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/meshname.txt 
         meta :                 (7,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/meta.txt 
     primname :              (3248,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/primname.txt 
      mmlabel :                (10,)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/mmlabel.txt 
         tran :         (8159, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/tran.npy 
         inst :        (48477, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/inst.npy 
        solid :           (10, 3, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/solid.npy 
         prim :         (3248, 4, 4)  : /tmp/blyth/opticks/ntds3_aug5/G4CXOpticks/CSGFoundry/prim.npy 
    """

    eprint("np.all( acf.node == bcf.node )", globals(), locals() )  
    eprint("np.all( acf.itra == bcf.itra )", globals(), locals() )
    eprint("np.all( acf.meshname == bcf.meshname )", globals(), locals() )
    #eprint("np.all( acf.meta == bcf.meta)" , globals(), locals() )
    eprint("np.all( acf.primname == bcf.primname )", globals(), locals() )
    eprint("np.all( acf.mmlabel == bcf.mmlabel )", globals(), locals() ) 
    eprint("np.all( acf.tran == bcf.tran )", globals(), locals() )
    eprint("np.all( acf.inst == bcf.inst )", globals(), locals() )
    eprint("np.all( acf.inst[:,:,:3] == bcf.inst[:,:,:3] ) ", globals(), locals(), tail=" exclude 4th column" )
    eprint("np.all( acf.solid == bcf.solid )", globals(), locals() )
    eprint("np.all( acf.prim == bcf.prim ) ", globals(), locals() )





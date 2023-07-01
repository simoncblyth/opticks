#!/usr/bin/env python

import numpy as np
import hashlib 
dig_ = lambda _:hashlib.md5(_).hexdigest()[:8]

from opticks.ana.fold import Fold

def reconstruct_mt(t):
    """ 
    :param t: Fold containing : optical, bd, bnd_names 
    :return mt:

    Vacuum ABSLENGTH of 1.e+09 within array triggers obnoxious array presentation 
    to make less obnoxious reduce that :   mt[16,0,:,1] = 1.e+06  
    """

    bd_ = "np.array( t.optical[:,0].reshape(-1,4), dtype=np.int32 ) - 1"
    bd = eval(bd_)

    print( bd_ )
    print( "bd.shape : %s " % str(bd.shape) )

    assert np.all( bd == t.bd )

    mbd = bd.copy() 
    mbd[:,1] = -1    # scrub the osur,isur pointers
    mbd[:,2] = -1  

    lines = t.bnd_names.lines
    name_bd = np.zeros( (len(lines), 4 ), dtype="|S50" )
    for i in range(len(name_bd)): name_bd[i] = lines[i].split("/")
    assert name_bd.shape == bd.shape


    num_mt = mbd.max()+1  
    mt_shape = (num_mt,) + t.bnd.shape[-3:] 

    mt = np.zeros( mt_shape, dtype=t.bnd.dtype ) 

    for i in range(num_mt):
        wr, wc = np.where( mbd == i )
        assert len(wr) == len(wc)
        assert len(wr) > 0 

        mt[i] = t.bnd[wr[0],wc[0]]
        dig   = dig_(mt[i].data)

        name = name_bd[wr[0],wc[0]]

        print(" i %d wr : %s wc %s dig %s name %s  " % ( i, str(wr), str(wc), dig, name ))  

        for j in range(len(wr)):
            other = t.bnd[wr[j],wc[j]]
            assert dig == dig_(other.data)
            other_name = name_bd[wr[j],wc[j]]
            assert name == other_name 
        pass
    return mt


if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))

    mt = reconstruct_mt(t)
    mt[16,0,:,1] = 1.e+06   # reduce Vacuum ABSLENGTH to avoid obnoxious presentation

    if np.all( t.mat[16,0,:,1] == 1e9 ):
        print("Vacuum 1e9 kludge reduce to 1e6 : because it causes obnoxious presentation")
        t.mat[16,0,:,1] = 1e6 
    else:
        print("Not doing Vacuum kludge")
    pass

    assert np.all( mt == t.mat )  # comparing the python reconstruct_mt with SBnd::mat_from_bd


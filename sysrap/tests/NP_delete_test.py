#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
import matplotlib.pyplot as mp

SIZE=np.array([1280, 720])


if __name__ == '__main__':
    f = Fold.Load(symbol="f")
    print(repr(f))

    rp = f.runprof
    assert rp.ndim == 2 and rp.shape[1] == 3 and rp.shape[0] > 0 

    st = ( rp[:,0] - rp[0,0])/1e6   # seconds
    vm = rp[:,1]/1e6   # GB
    rs = rp[:,2]/1e6   # GB 


    expr0 = "f.run/1e6 " 
    note0 = "%s : %s " % ( expr0, eval(expr0) )
    
    expr1 = "f.run*16*4/1e9 # GB estimate " 
    note1 = "%s : %s " % ( expr1, eval(expr1) )



    title_ = "~/opticks/sysrap/tests/NP_delete_test.sh TEST:%s CLEAR:%s DELETE:%s " % ( f.run_meta.TEST, f.run_meta.CLEAR, f.run_meta.DELETE  )  
    title = "\n".join([title_, note0, note1])

    fig, ax = mp.subplots(figsize=SIZE/100.)
    fig.suptitle(title)

    for idx in range(len(f.run)):
        fmt = "%0.3d" % idx 
        w = np.where( np.char.endswith( f.runprof_names, fmt ) )[0]  

        if "VM" in os.environ:
            ax.plot( st[w], vm[w], label="%s : VM[GB] vs time[s]" % fmt  ) 
        pass
        ax.plot( st[w], rs[w], label="%s : RSS[GB] vs time[s] " % fmt ) 
        ax.scatter( st[w], rs[w], label="%s : RSS[GB] vs time[s]" % fmt  ) 
    pass

    ax.legend()    
    fig.show()    
pass


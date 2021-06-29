#!/usr/bin/env python
"""
::

    ipython --pdb -i abplt.py 

"""
import numpy as np
from opticks.ana.ab import AB 
from opticks.ana.main import opticks_main

import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

qbins={}
qbins["wl"]=np.arange(300,600,5)
qbins["w0"]=np.arange(300,600,5)
qbins["t"] = np.arange(0, 300, 5) 
qbins["r"] = np.arange(0, 20100, 100) 

if __name__ == '__main__':
    ok = opticks_main()
    ab = AB(ok)  
    a = ab.a
    b = ab.b

    qq="w0 wl r t"
    #qq="w0"

    for q in qq.split():
        bins=qbins[q]

        ah = np.histogram( getattr(a, q), bins=bins )[0]
        bh = np.histogram( getattr(b, q), bins=bins )[0] 

        fig, ax = plt.subplots()
        ax.plot( bins[:-1], ah, label="OK:%s" % q, drawstyle="steps" )
        ax.plot( bins[:-1], bh, label="G4:%s" % q, drawstyle="steps" )
        ax.legend()
        fig.show()
    pass



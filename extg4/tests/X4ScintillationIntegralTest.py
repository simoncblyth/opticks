#!/usr/bin/env python
"""
X4ScintillationIntegralTest.py
===============================

::

    ipython -i tests/X4ScintillationIntegralTest.py

"""
import logging
log = logging.getLogger(__name__)
import json, numpy as np
import matplotlib.pyplot as plt 

class X4ScintillationIntegralTest(object):
    DIR="/tmp/G4OpticksAnaMgr" 
    NAME= "X4ScintillationIntegralTest_g4icdf.npy"
    def __init__(self):
        path = os.path.join(self.DIR,self.NAME) 
        self.icdf = np.load(path).reshape(3,-1)
        jspath = path.replace(".npy",".json")
        log.info("jspath:%s" % jspath)
        meta = json.load(open(jspath)) if os.path.exists(jspath) else {}
        for kv in meta.items():
            log.info(" %s : %s " % tuple(kv))
        pass
        self.meta = meta 
        self.hd_factor = float(meta.get("hd_factor", "10"))
        self.edge      = float(meta.get("edge", "0.1"))
    pass
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    xs = X4ScintillationIntegralTest()

    hd_factor = xs.hd_factor
    edge = xs.edge

    print("xs.icdf:%s  hd_factor:%s edge:%s " % (repr(xs.icdf.shape),hd_factor,edge))
    icdf = xs.icdf 

    i0 = icdf[0]
    i1 = icdf[1]
    i2 = icdf[2]

    fig, ax = plt.subplots()

    n = 4096  

    x0 = np.linspace(0, 1, n)

    num = 1000000
    u = np.random.rand(num)  

    lhs_ledge = 0.
    rhs_ledge = 1.-edge 

    ul = (u-lhs_ledge)*float(hd_factor)     # map 0.0->0.05 to 0->1
    ur = (u-rhs_ledge)*float(hd_factor)    # map 0.95->1.0 to 0->1

    wa = np.interp(u,  x0, i0 )   
    wl = np.interp(ul, x0, i1 )    
    wr = np.interp(ur, x0, i2 )   

    wa_lhs = np.interp(edge, x0, i0)
    wa_rhs = np.interp(rhs_ledge, x0, i0)

    s_mid = np.logical_and(u > edge, u < rhs_ledge) 
    s_lhs = u < edge
    s_rhs = u > rhs_ledge

    w = np.zeros(num)
    w[s_mid] = wa[s_mid]
    w[s_lhs] = wl[s_lhs] 
    w[s_rhs] = wr[s_rhs] 
    
    #ax = axs[0]
    #ax.plot( x0, i0, label="i0" )
    #ax.plot( x1, i1, label="i1" )
    #ax.plot( x2, i2, label="i2" )
    #ax.legend()

    bins = np.arange(300, 800, 1)  
    #ax = axs[1] 

    h,_ = np.histogram( w, bins )
    ha,_ = np.histogram( wa, bins )
    hl,_ = np.histogram( wl, bins )
    hr,_ = np.histogram( wr, bins )

    ax.plot( bins[:-1], h, label="h", drawstyle="steps")
    #ax.plot( bins[:-1], ha, label="ha", drawstyle="steps")
    #ax.plot( bins[:-1], hl, label="hl")
    #ax.plot( bins[:-1], hr, label="hr")

    ylim = np.array(ax.get_ylim())
    for x in [wa_lhs,wa_rhs]:
        ax.plot( [x,x], ylim*1.1 )    
    pass
    ax.legend()
    fig.show()


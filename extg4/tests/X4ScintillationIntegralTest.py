#!/usr/bin/env python
"""
X4ScintillationIntegralTest.py
===============================

::

    ipython -i tests/X4ScintillationIntegralTest.py

"""
import numpy as np
import matplotlib.pyplot as plt 

class X4ScintillationIntegralTest(object):
    DIR="/tmp/G4OpticksAnaMgr" 
    NAME= "X4ScintillationIntegralTest_icdf.npy"
    def __init__(self):
        self.icdf = np.load(os.path.join(self.DIR,self.NAME)).reshape(3,-1)
    pass

if __name__ == '__main__':
    xs = X4ScintillationIntegralTest()
    print("xs.icdf:%s " % repr(xs.icdf.shape))
    icdf = xs.icdf 

    i0 = icdf[0]
    i1 = icdf[1]
    i2 = icdf[2]

    fig, ax = plt.subplots()

    n = 4096  

    x0 = np.linspace(0, 1, n)


    num = 1000000
 
    u0 = np.random.rand(num)  
    u1 = (u0-0.0)*10.    # map 0.0->0.1 to 0->1
    u2 = (u0-0.9)*10.    # map 0.9->1.0 to 0->1

    w0 = np.interp(u0, x0, i0 )   
    w1 = np.interp(u1, x0, i1 )    
    w2 = np.interp(u2, x0, i2 )   

    w0_10 = np.interp(0.1, x0, i0)
    w0_90 = np.interp(0.9, x0, i0)

    s0 = np.logical_and(u > 0.1, u < 0.9) 
    s1 = u < 0.1
    s2 = u > 0.9

    w = np.zeros(num)
    w[s0] = w0[s0]
    w[s1] = w1[s1] 
    w[s2] = w2[s2] 
    
    #ax = axs[0]
    #ax.plot( x0, i0, label="i0" )
    #ax.plot( x1, i1, label="i1" )
    #ax.plot( x2, i2, label="i2" )
    #ax.legend()

    bins = np.arange(300, 800, 1)  
    #ax = axs[1] 

    h,_ = np.histogram( w, bins )
    h0,_ = np.histogram( w0, bins )
    h1,_ = np.histogram( w1, bins )
    h2,_ = np.histogram( w2, bins )

    ax.plot( bins[:-1], h, label="h", drawstyle="steps")
    #ax.plot( bins[:-1], h0, label="h0", drawstyle="steps")
    #ax.plot( bins[:-1], h1, label="h1")
    #ax.plot( bins[:-1], h2, label="h2")

    ylim = np.array(ax.get_ylim())
    for x in [w0_10,w0_90]:
        ax.plot( [x,x], ylim*1.1 )    
    pass
    ax.legend()
    fig.show()


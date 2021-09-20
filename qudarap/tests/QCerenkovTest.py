#!/usr/bin/env python
"""
QCerenkovTest.py
==================

::

    QCerenkovTest 
    ipython -i tests/QCerenkovTest.py


Getting very close match with symbolic integral results::


    In [10]: 1e12*(t.p_s2c[0,:,-1] - t.b_s2c[0,:,-1])                                                                                                                                                
    Out[10]: array([ 0.   ,  0.006,  0.009,  0.007,  0.007,  0.014,  0.018,  0.014,  0.028,  0.014,  0.   , -0.014,  0.028,  0.028,  0.028,  0.028,  0.057,  0.057])




"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

class QCerenkovTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/QCerenkovTest") 
    def __init__(self):
        pass

    def load(self, qwn, name):
        path = name if name[0] == "/" else os.path.join(self.FOLD, name)
        log.info("load %s " % path )
        a = np.load(path) if os.path.exists(path) else None
        if a is None:
            log.fatal("failed to load %s " % path )
        pass
        setattr(self, qwn, a )

    def load_getS2CumulativeIntegrals_many(self):
        self.load("a_s2c",  "test_getS2CumulativeIntegrals/s2c.npy")
        self.load("a_s2cn", "test_getS2CumulativeIntegrals/s2cn.npy")

    def load_getS2Integral_Cumulative_many(self):
        self.load("b_s2c",  "test_getS2Integral_Cumulative/s2c.npy")
        self.load("b_s2cn", "test_getS2Integral_Cumulative/s2cn.npy")

    def load_piecewise(self):
        self.load("p_s2c",  "/tmp/ana/piecewise/scan.npy")

    def plot(self, qwns, ii):
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        for qwn in qwns: 
            a = getattr(self, qwn)
            assert len(a.shape) == 3  
            for i in ii:
                ax.plot( a[i,:,0], a[i,:,-1] , label="%s  %d" % (qwn, i) )
            pass
            ax.legend()
        pass
        fig.show()

   
     
def plot_s2(t):

    en = t.s2c[0][:,0]
    s2 = t.s2c[0][:,1]

    fig, ax = plt.subplots(figsize=[12.8, 7.2])
    ax.plot( en, s2, label="s2" )
    ax.scatter( en, s2, label="s2" )
    ax.legend()  
    fig.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QCerenkovTest()

    #ii = np.arange( 0, 1000, 100 )
    ii = np.arange(9)

    #t.load_getS2CumulativeIntegrals_many()
    t.load_getS2Integral_Cumulative_many() 
    t.load_piecewise()

    #t.plot(["a_s2cn",], ii)
    #t.plot(["b_s2cn",], ii)

    t.plot(["b_s2c","p_s2c"], ii)

    #plot_s2(t)


    # with mul=1 this is giving excellent agreement less than 1e-12
    for i in ii:
        df = t.p_s2c[i,:,-1] - t.b_s2c[i,:,-1]
        print(df)
        print(df.max())
    pass 







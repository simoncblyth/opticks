#!/usr/bin/env python
"""
QCerenkovIntegralTest.py
============================

::

    QCerenkovIntegralTest 
    ipython -i tests/QCerenkovIntegralTest.py


Getting very close match with symbolic integral results::


    In [10]: 1e12*(t.p_s2c[0,:,-1] - t.b_s2c[0,:,-1])                                                                                                                                                
    Out[10]: array([ 0.   ,  0.006,  0.009,  0.007,  0.007,  0.014,  0.018,  0.014,  0.028,  0.014,  0.   , -0.014,  0.028,  0.028,  0.028,  0.028,  0.057,  0.057])


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

class QCerenkovIntegralTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/QCerenkovIntegralTest") 
    def __init__(self):
        pass

    def load(self, qwn, pfx, nam):
        name = os.path.join(pfx, nam) 
        path = name if name[0] == "/" else os.path.join(self.FOLD, name)
        log.debug("load %s " % path )
        a = np.load(path) if os.path.exists(path) else None
        if a is None:
            log.fatal("failed to load %s " % path )
        else:
            os.system("ls -l %s" % path )
        pass
        setattr(self, qwn, a )

    def load_getS2Integral_UpperCut(self):
        pfx = "test_getS2Integral_UpperCut"
        self.load("a_s2c",  pfx, "s2c.npy")
        self.load("a_s2cn", pfx, "s2cn.npy")

    def load_getS2Integral_SplitBin(self):
        pfx = "test_getS2Integral_SplitBin"
        self.load("b_s2c",  pfx, "s2c.npy")
        self.load("b_s2cn", pfx, "s2cn.npy")
        self.load("b_bis",  pfx, "bis.npy")

    def load_piecewise(self):
        pfx = "/tmp/ana/piecewise"  
        self.load("p_s2c", pfx, "scan.npy")
        self.load("p_bis", pfx, "bis.npy")

    def plot(self, qwns, ii):
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        for qwn in qwns: 
            a = getattr(self, qwn)
            #ken = 1 if qwn[0] == "b" else 0 
            ken = 0   # reajjanged to put en_b into payload slot 0 
            log.debug("qwn %s shape %s ken %d " % (qwn, str(a.shape), ken)) 
            assert len(a.shape) == 3  
            for i in ii:
                if qwn[0] == "b":
                    ax.scatter( a[i,:,ken], a[i,:,-1] , label="%s  %d" % (qwn, i) )
                else:
                    ax.plot( a[i,:,ken], a[i,:,-1] , label="%s  %d" % (qwn, i) )
                pass
            pass
            ax.legend()
        pass
        fig.show()


    def compare(self, ii, sa, sb, ):
        """
        * with mul=1 this is giving excellent agreement less than 1e-12
        * huh, after adding SUB handling discrep up to almost 0.1 photon
        """
        t = self
        a = getattr(t, sa)
        b = getattr(t, sb)
        log.info(" sa:%s a:%s sb:%s b:%s " % (sa, str(a.shape), sb, str(b.shape)))
        for i in ii:
            BetaInverse = t.b_bis[i]
            df = np.abs(a[i,:,-1] - b[i,:,-1])
            dfmax = df.max()

            #print("BetaInverse : %10.4f  dfmax %10.4g  df %s " % (BetaInverse, dfmax, str(df))  )
            print("BetaInverse : %10.4f  dfmax %10.4g " % (BetaInverse, dfmax )  )
        pass 

   
     
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
    t = QCerenkovIntegralTest()

    #ii = np.arange( 0, 1000, 100 )
    ii = np.arange(9)
    #ii = [0]

    #t.load_getS2Integral_UpperCut()
    t.load_getS2Integral_SplitBin() 
    t.load_piecewise()

    #t.plot(["a_s2cn",], ii)
    #t.plot(["b_s2cn",], ii)

    t.plot(["b_s2c","p_s2c"], ii)

    #plot_s2(t)

    assert np.all( t.p_bis == t.b_bis )
    t.compare(ii, "p_s2c", "b_s2c")





#!/usr/bin/env python
"""
QCKTest.py
==================

::

    QCerenkovTest 
    ipython -i tests/QCKTest.py


See also::

    ana/rindex.py 
    ana/ckn.py 

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)


class QCKTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/QCerenkovTest") 
    def __init__(self):
        base_path = os.path.join(self.FOLD, "test_makeICDF")
        if not os.path.exists(base_path):
            log.fatal("base_path %s does not exist" % base_path)
            return 
        pass 
        names = os.listdir(base_path)
        log.info("loading from base_path %s " % base_path)
        for name in filter(lambda _:_.endswith(".npy"), names):
            path = os.path.join(base_path, name)
            stem = name[:-4]
            a = np.load(path) 
            print( " t.%5s  %s " % (stem, str(a.shape))) 
            setattr(self, stem, a )
        pass
    pass

    def s2cn_plot(self, ii):
        """
        :param ii: list of first dimension indices, corresponding to BetaInverse values
        """
        s2cn = self.s2cn
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        for i in ii:
            ax.plot( s2cn[i,:,0], s2cn[i,:,1] , label="%d" % i )
        pass
        #ax.legend()
        fig.show()


    def en_plot(self):
        en = np.load(os.path.expandvars("$TMP/QCKTest/test_energy_lookup_many/en.npy"))
        self.en = en 
        lo = 1.55
        hi = 15.5 

        h = np.histogram( en, bins=np.linspace(lo,hi,100))
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        ax.scatter( h[1][:-1], h[0] )
        fig.show() 
        








if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QCKTest()
    ii = np.arange( 0, 1000, 10 )
    t.s2cn_plot(ii)
    t.en_plot() 








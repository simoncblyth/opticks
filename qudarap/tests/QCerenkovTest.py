#!/usr/bin/env python
"""
QCerenkovTest.py
==================

::

    QCerenkovTest 
    ipython -i tests/QCerenkovTest.py


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

class QCerenkovTest(object):
    FOLD = os.path.expandvars("/tmp/$USER/opticks/QCerenkovTest") 
    def __init__(self):
        s2c_path = os.path.join(self.FOLD, "test_getS2CumulativeIntegrals_many_s2c.npy")
        s2c = np.load(s2c_path) if os.path.exists(s2c_path) else None
        if s2c is None:
            log.error("recreate input arrays by running : QCerenkovTest" ); 
        pass
        self.s2c = s2c


        s2cn_path = os.path.join(self.FOLD, "test_getS2CumulativeIntegrals_many_s2cn.npy")
        self.s2cn = np.load(s2cn_path) if os.path.exists(s2cn_path) else None
    pass

    def s2c_plot(self, ii):
        s2cn = self.s2cn
        fig, ax = plt.subplots(figsize=[12.8, 7.2])
        for i in ii:
            ax.plot( s2cn[i,:,0], s2cn[i,:,1] , label="%d" % i )
        pass
        #ax.legend()
        fig.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QCerenkovTest()

    ii = np.arange( 0, 1000, 10 )

    t.s2c_plot(ii)



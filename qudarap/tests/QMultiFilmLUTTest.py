#!/usr/bin/env python

"""
QMultiFilmLUTTest.py
================

::

    QMultiFilmLUTTest 
    ipython -i tests/QMultiFilmLUTTest.py

"""
import os, logging, numpy as np
log = logging.getLogger(__name__)
#from opticks.ana.nload import stamp_

import matplotlib.pyplot as plt
import matplotlib.cm  as cm

class QMultiFilmLUTTest(object):
    BASE = os.path.expandvars("/tmp/$USER/opticks/QMultiFilmLUTTest") 
    def __init__(self):
        path = os.path.join(self.BASE,"src.npy")
        self.src = np.load(path)
        pass
    def test_all_lookup(self):
        src = self.src
        for pmtcatIdx in range(src.shape[0]):
            for bndIdx in range(src.shape[1]):
                for resIdx in range(src.shape[2]):
                    dst_file_name = "pmtcat_{}bnd_{}resolution_{}.npy".format(pmtcatIdx,bndIdx,resIdx)
                    self.subsrc = self.src[pmtcatIdx, bndIdx ,resIdx ,:,:,:]
                    self.test_lookup(dst_file_name)
                    
    def test_lookup(self, dst_file_name):
        path = os.path.join(self.BASE,dst_file_name)
        dst = np.load(path)
        print("dst.shape = {} dst = {} , src.shape = {}".format(dst.shape, dst , self.src.shape))
        
        subsrc = self.subsrc
        fig, axs = plt.subplots(3,4)
        #for i in range(2):
        for j in range(4):
            plt.cla()
            origin = subsrc[:,:,j]
            gen = dst[:,:,j]
            diff = gen - origin
            imx = axs[0,j].imshow(origin)
            plt.colorbar(mappable=imx, ax=axs[0,j])
            imy = axs[1,j].imshow(gen) 
            plt.colorbar(mappable=imy, ax=axs[1,j])
            imz = axs[2,j].imshow(diff)
            plt.colorbar(mappable=imz, ax=axs[2,j])
        #fig.show()
        #fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs)
        plt.suptitle(dst_file_name + "\n Rs_Ts_Rp_Tp",fontsize = 30 )
        plt.show()

        '''
        fold = os.path.join(self.BASE, "test_lookup")
        names = os.listdir(fold)
        for name in filter(lambda n:n.endswith(".npy"),names):
            path = os.path.join(fold, name)
            stem = name[:-4]
            a = np.load(path)
            log.info(" %10s : %20s : %s : %s " % ( stem, str(a.shape), stamp_(path), path )) 
            setattr( self, stem, a ) 
            globals()[stem] = a 
        pass
        assert np.all( icdf_src == icdf_dst )  
        '''

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QMultiFilmLUTTest()
    t.test_all_lookup()

    



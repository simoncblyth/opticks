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
import matplotlib as mpl
class QMultiFilmTest(object):
    BASE = os.path.expandvars("/tmp/$USER/opticks/QMultiFilmTest") 
    def __init__(self):
        mpl.rcParams['font.size'] = 15
        path = os.path.join(self.BASE,"src.npy")
        self.src = np.load(path)
        pass
    def test_all_lookup(self):
        src = self.src
        for pmtcatIdx in range(src.shape[0]):
            for resIdx in range(src.shape[1]):
                dst_file_name = "pmtcat_{}resolution_{}.npy".format(pmtcatIdx,resIdx)
                self.subsrc = self.src[pmtcatIdx,resIdx ,:,:,:]
                self.test_lookup(dst_file_name)         
	        
    def test_lookup(self, dst_file_name):
        path = os.path.join(self.BASE,dst_file_name)
        dst = np.load(path)
        print("dst.shape = {} dst = {} , src.shape = {}".format(dst.shape, dst , self.src.shape))
        
        subsrc = self.subsrc
        fig, axs = plt.subplots(3,4,figsize=(12,8))
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
        plt.suptitle(dst_file_name + " Rs_Ts_Rp_Tp",fontsize = 16 )
        plt.show()

    def test_mock_lookup(self):
        path = os.path.join(self.BASE,"test_texture.npy")
        input_arr = np.load(path)
        print("input_arr shape = {}".format(input_arr.shape))
        
        path = os.path.join(self.BASE,"gpu_texture_interp.npy")
        res_arr = np.load(path)
        print("res_arr shape = {}".format(res_arr.shape))
      
         
        a_Rs = input_arr[:,:,1,0]  
        a_Ts = input_arr[:,:,1,1]  
        a_Rp = input_arr[:,:,1,2]  
        a_Tp = input_arr[:,:,1,3]  
        b_Rs = res_arr[:,:,0]
        b_Ts = res_arr[:,:,1]
        b_Rp = res_arr[:,:,2]
        b_Tp = res_arr[:,:,3]

        fig,ax = plt.subplots( figsize=(8,6) )
        ax.scatter(a_Rs,a_Rs - b_Rs,label= r"$R_s$",s=4,color='b')
        ax.scatter(a_Ts,a_Ts - b_Ts,label= r"$T_s$",s=4,color='g')
        ax.scatter(a_Rp,a_Rp - b_Rp,label= r"$R_p$",s=4,color='r')
        ax.scatter(a_Tp,a_Tp - b_Tp,label= r"$T_p$",s=4,color='m')

        ax.legend()	
        ax.set_xlabel("Calculation value")
        ax.set_ylabel("Difference")
        ax.tick_params(left= True, bottom = True, right =True, top= True, which = "both", direction = "in")
        ax.tick_params(left= True, bottom = True, right =True, top=True, which = "both", width = 1.5)
        ax.tick_params(left= True, bottom = True, right =True, top= True, which = "minor", length = 6)
        ax.tick_params(left= True, bottom = True, right =True, top= True, which = "major", length = 12)
        ax.minorticks_on()
        
        ax.grid(axis="both",linestyle="--")
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        
        plt.show()


        fig,ax = plt.subplots( figsize=(8,6) )
        bins = np.logspace(-5,0,50)
        ax.hist((a_Rs - b_Rs).flatten(), bins, label= r"$R_s$",histtype ="step", color='b',linewidth = 2)
        ax.hist((a_Ts - b_Ts).flatten(), bins, label= r"$T_s$",histtype ="step", color='g',linewidth = 2)
        ax.hist((a_Rp - b_Rp).flatten(), bins, label= r"$R_p$",histtype ="step", color='r',linewidth = 2)
        ax.hist((a_Tp - b_Tp).flatten(), bins, label= r"$T_p$",histtype ="step", color='m',linewidth = 2)
        ax.legend()	
        ax.set_xlabel("Difference")
        ax.set_xscale("log")
        ax.set_ylabel("Entry")
        ax.set_yscale("log")
        ax.tick_params(left= True, bottom = True, right =True, top= True, which = "both", direction = "in")
        ax.tick_params(left= True, bottom = True, right =True, top=True, which = "both", width = 1.5)
        ax.tick_params(left= True, bottom = True, right =True, top= True, which = "minor", length = 6)
        ax.tick_params(left= True, bottom = True, right =True, top= True, which = "major", length = 12)
        ax.minorticks_on()
        
        ax.grid(axis="both",linestyle="--")
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        
        plt.show()
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = QMultiFilmTest()
    #t.test_all_lookup()
    t.test_mock_lookup()

    



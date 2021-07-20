#!/usr/bin/env python
"""
wavelength.py
===============

::

   ARG=6 ipython -i wavelength.py 
   ARG=7 ipython -i wavelength.py 
   ARG=8 ipython -i wavelength.py 
   ARG=11 ipython -i wavelength.py 


"""

import os, numpy as np, logging
log = logging.getLogger(__name__)
from opticks.ana.key import keydir

class Wavelength(object):
    """
    Comparing wavelength distribs between many different samples
    """
    def get_key(self, label):
        key = None 
        for k,v in self.l.items(): 
            if v == label:
                key = k
            pass
        pass
        return key 

    def get_keys(self, a_label, b_label):
        a = self.get_key(a_label)
        b = self.get_key(b_label)
        return a, b 

    def __call__(self, label):
        return self.get_key(label) 

    def __init__(self, kd):
        p = {}
        l = {}

        l[0] = "DsG4Scintillator_G4OpticksAnaMgr"    ## horses mouth
        p[0] = "/tmp/G4OpticksAnaMgr/WavelengthSamples.npy"

        l[1] = "Opticks_QCtxTest_hd20"
        p[1] = os.path.join("/tmp/QCtxTest", "wavelength_20.npy")

        l[2] = "Opticks_QCtxTest_hd0"
        p[2] = os.path.join("/tmp/QCtxTest", "wavelength_0.npy")

        l[3] = "Opticks_QCtxTest_hd20_cudaFilterModePoint"
        p[3] = os.path.join("/tmp/QCtxTest", "wavelength_20_cudaFilterModePoint.npy")

        l[4] = "Opticks_QCtxTest_hd0_cudaFilterModePoint"
        p[4] = os.path.join("/tmp/QCtxTest", "wavelength_0_cudaFilterModePoint.npy")

        l[5] = "X4"
        p[5] = "/tmp/X4ScintillationTest/g4localSamples.npy"

        l[6] = "GScintillatorLib_np_interp"
        p[6] = os.path.join(kd,"GScintillatorLib/GScintillatorLib.npy") 

        l[7] = "ck_photon_1M"
        p[7] = os.path.join("/tmp/QCtxTest", "cerenkov_photon_1000000.npy")
      
        l[8] = "G4Cerenkov_modified_SKIP_CONTINUE"
        p[8] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_step_length_100000.000_SKIP_CONTINUE", "GenWavelength.npy")

        l[9] = "G4Cerenkov_modified_ASIS"
        p[9] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_step_length_100000.000_ASIS", "GenWavelength.npy")

        l[10] = "G4Cerenkov_modified_SKIP_CONTINUE_10k"
        p[10] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_override_fNumPhotons_10000_SKIP_CONTINUE", "GenWavelength.npy")

        l[11] = "G4Cerenkov_modified_SKIP_CONTINUE_1M"
        p[11] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUE", "GenWavelength.npy")

        l[12] = "G4Cerenkov_modified_SKIP_CONTINUE_1M_seed1" 
        p[12] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUEseed_1_", "GenWavelength.npy")

        l[13] = "G4Cerenkov_modified_SKIP_CONTINUE_1M_seed2" 
        p[13] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUEseed_2_", "GenWavelength.npy")

        l[14] = "G4Cerenkov_modified_SKIP_CONTINUE_1M_FLOAT_TEST"
        p[14] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUE_FLOAT_TEST", "GenWavelength.npy")

        l[15] = "ck_photon_1M_FLIP_RANDOM"
        p[15] = os.path.join("/tmp/QCtxTest", "cerenkov_photon_FLIP_RANDOM_1000000.npy")
 
        l[16] = "G4Cerenkov_modified_SKIP_CONTINUE_1M_seed1f" 
        p[16] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUE_FLOAT_TEST_seed_1_", "GenWavelength.npy")

        l[17] = "G4Cerenkov_modified_SKIP_CONTINUE_1M_seed2f" 
        p[17] = os.path.join("/tmp/G4Cerenkov_modifiedTest", "BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUE_FLOAT_TEST_seed_2_", "GenWavelength.npy")

        l[18] = "ana_ck_1M"
        p[18] = "/tmp/ck/ck_1000000.npy"

        l[19] = "G4Cerenkov_modified_SKIP_CONTINUE_1M_PRECOOKED"
        p[19] = "/tmp/G4Cerenkov_modifiedTest/BetaInverse_1.500_override_fNumPhotons_1000000_SKIP_CONTINUE_PRECOOKED/GenWavelength.npy"


        dom = np.arange(80, 400, 4)  
        #dom = np.arange(300, 600, 1)  
        #dom = np.arange(385, 475, 1)  


        #dom = np.arange(350, 550, 1)  


        a = {}
        w = {}
        h = {}
        for i in range(len(l)):
            if not os.path.exists(p[i]):
                a[i] = None
                w[i] = None
                h[i] = None
            else:
                a[i] = np.load(p[i])
                if l[i].startswith("ck_photon"):
                    w[i] = a[i][:,0,1] 
                elif l[i].startswith("ana_ck"):
                    w[i] = a[i][:,0,1] 
                elif l[i].startswith("G4Cerenkov_modified"):
                    w[i] = a[i][:,0,1] 
                elif l[i] == "GScintillatorLib_np_interp":
                    aa = a[i] 
                    self.aa = aa
                    aa0 = aa[0,:,0]
                    bb0 = np.linspace(0,1,len(aa0))
                    u = np.random.rand(1000000)  
                    w[i] = np.interp(u, bb0, aa0 )  
                else:
                    w[i] = a[i]
                pass
                h[i] = np.histogram( w[i] , dom ) 
            pass
        pass
        self.p = p  
        self.w = w  
        self.l = l
        self.h = h   
        self.a = a   
        self.dom = dom 

    def interp(self, u):
        a = self.aa[0,:,0]
        b = np.linspace(0,1,len(a))
        return np.interp( u, b, a )


    def cf(self, arg):
        if arg == 0:
            a, b = self.get_keys('DsG4Scintillator_G4OpticksAnaMgr', "Opticks_QCtxTest_hd20") 
        elif arg == 1:
            a, b = self.get_keys('DsG4Scintillator_G4OpticksAnaMgr', "Opticks_QCtxTest_hd0") 
        elif arg == 2:
            a, b = self.get_keys('DsG4Scintillator_G4OpticksAnaMgr', 'Opticks_QCtxTest_hd20_cudaFilterModePoint') 
        elif arg == 3:
            a, b = self.get_keys('DsG4Scintillator_G4OpticksAnaMgr', 'Opticks_QCtxTest_hd0_cudaFilterModePoint')
        elif arg == 4:
            a, b = self.get_keys('G4Cerenkov_modified_SKIP_CONTINUE', 'ck_photon' )
        elif arg == 5:
            a, b = self.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_10k', 'ck_photon_10k' )
        elif arg == 6:
            a, b = self.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_1M', 'ck_photon_1M' )
        elif arg == 7:
            a, b = self.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_1M_seed1', 'G4Cerenkov_modified_SKIP_CONTINUE_1M_seed2' )
        elif arg == 8:
            a, b = self.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_1M_FLOAT_TEST', 'ck_photon_1M' )
        elif arg == 9:
            a, b = self.get_keys('ck_photon_1M', 'ck_photon_1M_FLIP_RANDOM' )
        elif arg == 10:
            a, b = self.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_1M_seed1f', 'G4Cerenkov_modified_SKIP_CONTINUE_1M_seed2f' )
        elif arg == 11:
            a, b = self.get_keys('G4Cerenkov_modified_SKIP_CONTINUE_1M_PRECOOKED', 'ana_ck_1M' )
        else:
            assert 0
        pass
        return a, b



if __name__ == '__main__':
    kd = keydir(os.environ["OPTICKS_KEY"])
    wl = Wavelength(kd)
    arg = int(os.environ.get("ARG","0")) 
    ia,ib = wl.cf(arg)

    pa = wl.p[ia] 
    pb = wl.p[ib] 

    a = wl.a[ia]
    b = wl.a[ib]

    wa = wl.w[ia] 
    wb = wl.w[ib]
    la = wl.l[ia]
    lb = wl.l[ib]

    print("la:%s" % la)
    print("pa:%s" % pa)
    print("lb:%s" % lb)
    print("pb:%s" % pb)


    dev = np.abs( wa - wb ) > 1e-4
    num_dev = np.count_nonzero(dev) 
    print("num_dev:%d " % num_dev )

    np.set_printoptions(edgeitems=16)    

    print("a[dev]\n",a[dev].reshape(-1,8))
    print("b[dev]\n",b[dev].reshape(-1,16))
  
    b_ri = b[dev][:,0,2] 

    # are deviants mostly where sampled rindex bin values 
    # TODO: enable dumping of deviants, to understand the reason  

    fig, ax = plt.subplots() 
    ax.hist( b_ri, bins=50 )  
    fig.show() 
 
    #mask = np.where(dev)[0]    
    #np.save("/tmp/wavelength_deviant_mask.npy", mask) 
 



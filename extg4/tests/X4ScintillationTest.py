#!/usr/bin/env python
"""
X4ScintillationTest.py
===============================

::

    ipython -i tests/X4ScintillationTest.py

"""
import logging, os, subprocess
log = logging.getLogger(__name__)
import json, numpy as np
# dont import matplotlib at top level as it fails remotely, do it in the _plt method

G4_VERSION = int(subprocess.getoutput("Geant4VersionInteger"))
CLHEP_VERSION = int(subprocess.getoutput("CLHEPVersionInteger"))

class X4ScintillationTest(object):
    DIR=os.path.expandvars("$TMP/X4ScintillationTest")
    NAME= "g4icdf_auto.npy"
    LIBNAME = "GScintillatorLib.npy" 
    ENERGY_NAME= "g4icdf_energy_manual.npy"

    def __init__(self):
        icdf0_path = os.path.join(self.DIR, self.LIBNAME )
        icdf0 = np.load(icdf0_path).reshape(3,-1)
        self.icdf0_path = icdf0_path
        self.icdf0 = icdf0

        icdf_path = os.path.join(self.DIR,self.NAME) 
        icdf = np.load(icdf_path).reshape(3,-1)
        self.icdf_path = icdf_path
        self.icdf = icdf

        energy_icdf = np.load(os.path.join(self.DIR,"g4icdf_energy_manual.npy")).reshape(3,-1)
        self.energy_icdf = energy_icdf

        icdf_jspath = icdf_path.replace(".npy",".json")
        log.info("icdf_jspath:%s" % icdf_jspath)
        meta = json.load(open(icdf_jspath)) if os.path.exists(icdf_jspath) else {}
        for kv in meta.items():
            log.info(" %s : %s " % tuple(kv))
        pass
        self.meta = meta 
        self.hd_factor = float(meta.get("hd_factor", "10"))
        self.edge      = float(meta.get("edge", "0.1"))
        self.constants = np.load(os.path.expandvars("$TMP/X4PhysicalConstantsTest/%d.npy" % G4_VERSION ))
        self.constants_txt = np.loadtxt(os.path.expandvars("$TMP/X4PhysicalConstantsTest/%d.txt" % G4_VERSION ), dtype=np.object)

    def icdf_compare(self):
        a = self.icdf0
        b = self.icdf
        ab = np.abs(a-b) 
        log.info("icdf_compare")
        print("a:%s a.min %10g a.max %10g" % (str(a.shape),a.min(), a.max()))
        print("b.%s b.min %10g b.max %10g" % (str(b.shape),b.min(), b.max()))
        print("ab:%s ab.min %10g ab.max %10g" % (str(ab.shape),ab.min(), ab.max()))

        qwns = "a b ab"
        for qwn in qwns.split():
            globals()[qwn] = locals()[qwn] 
            setattr(self, qwn, locals()[qwn])
        pass

    def e2w_reciprocal_check(self):
        fac = self.constants[3]   #  G4:1042 CLHEP:2451  'h_Planck*c_light/nm'  0.0012398419843320022  1/806.5543937349214   
        energy_icdf = self.energy_icdf
        icdf = self.icdf

        wl_icdf = fac/energy_icdf
        ab = np.abs(wl_icdf - t.icdf)
        abmax = ab.max()
        print("compare wl_icdf obtained from energy_icdf from standard(wavelength) icdf" ) 
        print("abmax:%20g "  % abmax)

        
        qwns = "wl_icdf"
        for qwn in qwns.split():
            globals()[qwn] = locals()[qwn] 
        pass 


    def interp_plt(self):
        t = self
        hd_factor = t.hd_factor
        edge = t.edge

        print("t.icdf:%s  hd_factor:%s edge:%s " % (repr(t.icdf.shape),hd_factor,edge))
        icdf = t.icdf 

        i0 = icdf[0]
        i1 = icdf[1]
        i2 = icdf[2]

        import matplotlib.pyplot as plt 
        plt.ion()

        fig, ax = plt.subplots(figsize=[12.8, 7.2])
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

    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = X4ScintillationTest()
    t.icdf_compare()
    t.e2w_reciprocal_check()

    #t.interp_plt()






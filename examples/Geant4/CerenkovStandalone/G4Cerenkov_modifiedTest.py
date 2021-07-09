#!/usr/bin/env python
"""

::

    ipython -i G4Cerenkov_modifiedTest.py

"""
import os, logging, numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict as odict
log = logging.getLogger(__name__)

#np.set_printoptions(suppress=False, precision=8)   
np.set_printoptions(suppress=True, precision=4) 

names_ = lambda path:list(map(str.strip,open(path).readlines()))


class OpticksDebug(object): 
    def __init__(self, fold, name):
        self.load(fold, name)
    def load(self, fold, name):
        a = np.load(os.path.join(fold, "%s.npy" % name))
        n = names_(os.path.join(fold, "%s.txt" % name))
        i = np.array(n).reshape(-1,4) 
        self.a = a      
        self.n = n 
        self.i = i

    def __call__(self, name):
        a = self.a
        i = self.i
        jk = np.where( i == name )
        j = jk[0][0]  
        k = jk[1][0]
        q = a[:,j,k]
        return q 



class G4Cerenkov_modifiedTest(object):
    FOLD = "/tmp/G4Cerenkov_modifiedTest" 
    NAMES = "RINDEX.npy"

    @classmethod
    def RelDirs(cls):
        all_reldirs = list(filter(lambda n:n.startswith("BetaInverse_"),os.listdir(cls.FOLD)))
        reldirs = []
        for reldir in all_reldirs:
            missing = 0 
            for name in cls.NAMES.split():
                path=os.path.join(cls.FOLD, reldir, name)
                if not os.path.exists(path):
                    missing += 1 
                pass
            pass
            if missing == 0:
                reldirs.append(reldir)
            else:
                log.info("skip reldir %s " % reldir )
            pass 
        pass          
        return reldirs

    def __init__(self, reldir): 
        self.dir = os.path.join(self.FOLD, reldir)
        self.rindex = np.load(os.path.join(self.dir,  "RINDEX.npy"))  
        self.pho = np.load(os.path.join(self.dir,"photons.npy"))
        self.par = OpticksDebug(self.dir, "Params")
        self.gen = OpticksDebug(self.dir, "GenWavelength")
 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    reldirs = G4Cerenkov_modifiedTest.RelDirs()
    tt = odict()
    for reldir in reldirs:
        log.info(reldir) 
        t = G4Cerenkov_modifiedTest(reldir)
        tt[reldir] = t
    pass


for t in tt.values():

    p = t.pho 
    a = t.gen.a 
    n = t.gen.n 
    i = t.gen.i 

    BetaInverse = t.par('BetaInverse')[0]  
    beta = t.par('beta')[0]
    Pmin = t.par('Pmin')[0]*1e6
    Pmax = t.par('Pmax')[0]*1e6
    nMax = t.par('nMax')[0]
    maxCos = t.par('maxCos')[0]
    maxSin2 = t.par('maxSin2')[0]
    fNumPhotons = t.par('fNumPhotons')[0]

    qwn = "BetaInverse beta Pmin Pmax nMax maxCos maxSin2 fNumPhotons".split()
    fmt = " ".join(map(lambda q:"%s %%7.3f" % q, qwn)) 
    val = tuple(map(lambda q:globals()[q], qwn )) 
    title_qwn = fmt % val
    title = "\n".join([t.dir, title_qwn])
    print(title)

    rindex = t.rindex

    e = t.gen("sampledEnergy")*1e6 
    w = t.gen("sampledWavelength") 
    r = t.gen("sampledRI")  
    c = t.gen("cosTheta") 

    s2 = t.gen("sin2Theta")
    bi = t.gen("BetaInverse") 
    ht = t.gen("head_tail").copy().view(np.uint32).reshape(-1,2)
    cc = t.gen("continue_condition").copy().view(np.uint32).reshape(-1,2)

    nbin = 100 
    e_dom = np.linspace(Pmin, Pmax, nbin)
    w_dom = np.linspace(1240./Pmax, 1240./Pmin/2, nbin)
    r_dom = np.linspace(1.5, 1.8, nbin)
    c_dom = np.linspace(0.83, 1., nbin)
    s2_dom = np.linspace(0, 0.3, nbin) 

    h_e = np.histogram(e, e_dom)
    h_w = np.histogram(w, w_dom)
    h_r = np.histogram(r, r_dom)
    h_c = np.histogram(c, c_dom)
    h_s2 = np.histogram(s2, s2_dom)

    nrows = 2
    ncols = 3 
    figsize = [12.8, 7.2]
    fig, axs = plt.subplots(nrows,ncols, figsize=figsize)

    fig.suptitle(title)


    ax = axs[0,0]
    ax.plot( h_e[1][:-1], h_e[0], label="e", drawstyle="steps" )
    ax.set_ylim( 0, h_e[0].max()*1.1 )
    ylim = ax.get_ylim() 
    ax.plot( [Pmin, Pmin], ylim,  color="r", linestyle="dashed" )
    ax.plot( [Pmax, Pmax], ylim,  color="r", linestyle="dashed" )
    ax.legend()

    ax = axs[0,1]
    ax.plot( h_w[1][:-1], h_w[0], label="w", drawstyle="steps" )
    ax.set_ylim( 0, h_w[0].max()*1.1 )
    ax.legend()

    ax = axs[0,2]
    ax.plot( h_r[1][:-1], h_r[0], label="ri", drawstyle="steps" )
    ax.set_ylim( 0, h_r[0].max()*1.1 )
    ylim = ax.get_ylim() 
    ax.plot( [nMax, nMax], ylim,  color="r", linestyle="dashed" )
    ax.legend()

    ax = axs[1,0]
    ax.plot( h_c[1][:-1], h_c[0], label="c", drawstyle="steps" )
    ax.set_ylim( 0, h_c[0].max()*1.1 )

    h_c_imx = np.argmax(h_c[0])
    h_c_vmx = h_c[1][h_c_imx]
    h_c_vmx_s2 = (1.-h_c_vmx)*(1.+h_c_vmx)  

    ylim = ax.get_ylim() 
    ax.plot( [maxCos, maxCos], ylim,  color="r", linestyle="dashed" )
    ax.legend()

    ax = axs[1,1]
    ax.plot( h_s2[1][:-1], h_s2[0], label="s2", drawstyle="steps" )
    ax.set_ylim( 0, h_s2[0].max()*1.1 )
    ylim = ax.get_ylim() 
    ax.plot( [maxSin2, maxSin2], ylim,  color="r", linestyle="dashed" )
    ax.legend()

    h_s2_imx = np.argmax(h_s2[0])
    h_s2_vmx = h_s2[1][h_s2_imx]





    ax = axs[1,2]
    ax.plot( rindex[:,0]*1e6, rindex[:,1], label="rindex", drawstyle="steps" )
    xlim = ax.get_xlim() 
    ax.plot( xlim, [nMax, nMax], color="r", linestyle="dashed" )
    ax.plot( xlim, [BetaInverse, BetaInverse], color="b", linestyle="dashed" )

    ax.set_ylim( 1, 2 )
    ax.legend()

    fig.show()


    sBetaInverse = str(BetaInverse).replace(".","p")
    path=os.path.join(t.dir, "pngs", "BetaInverse_%s.png" % sBetaInverse ) 
    fold=os.path.dirname(path)
    if not os.path.isdir(fold):
       os.makedirs(fold)
    pass 
    log.info("save to %s " % path)
    fig.savefig(path)


    for i in range(10): print(" head:%2d tail:%2d continue:%2d condition:%2d cosTheta %10.3f  " % ( ht[i,0],ht[i,1],cc[i,0],cc[i,1], c[i] ))                                                     




"""
 head: 2 tail: 1 continue: 1 condition: 2 cosTheta      0.918  
 head: 9 tail: 2 continue: 7 condition: 9 cosTheta      0.994  
 head: 1 tail: 1 continue: 0 condition: 1 cosTheta      0.896  
 head: 1 tail: 0 continue: 1 condition: 1 cosTheta      0.896  
 head: 1 tail: 0 continue: 1 condition: 1 cosTheta      0.896  
 head: 1 tail: 0 continue: 1 condition: 1 cosTheta      0.896  
 head: 1 tail: 0 continue: 1 condition: 1 cosTheta      0.896  
 head: 1 tail: 1 continue: 0 condition: 1 cosTheta      0.897  
 head: 1 tail: 0 continue: 1 condition: 1 cosTheta      0.897  
 head:12 tail: 3 continue: 9 condition:12 cosTheta      1.000  


In [17]: np.where( cc[:,0] == 1 )                                                                                                                                                                    
Out[17]: (array([   0,    3,    4, ..., 7309, 7310, 7311]),)

In [18]: sel =  cc[:,0] == 1                                                                                                                                                                         

In [19]: cc[:,1][sel]                                                                                                                                                                                
Out[19]: array([2, 1, 1, ..., 1, 1, 1], dtype=uint32)

In [20]: ht[:,1][sel]                                                                                                                                                                                
Out[20]: array([1, 0, 0, ..., 0, 0, 0], dtype=uint32)

In [21]: np.count_nonzero(ht[:,1][sel] == 0)                                                                                                                                                         
Out[21]: 4294

In [22]: cc[:,1][sel].shape                                                                                                                                                                          
Out[22]: (4670,)

In [23]: cc.shape                                                                                                                                                                                    
Out[23]: (7312, 2)



"""



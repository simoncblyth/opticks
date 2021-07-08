#/usr/bin/env python
"""
::

     ipython -i L4CerenkovTest.py 

"""
import os, numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
pass

np.set_printoptions(suppress=False, precision=8)   

names_ = lambda path:list(map(str.strip,open(path).readlines()))


class L4CerenkovTest(object):
    DIR = "/tmp/L4CerenkovTest"
    def load(self, name):
        a = np.load(os.path.join(self.DIR, "%s.npy" % name))
        n = names_(os.path.join(self.DIR, "%s.txt" % name))
        i = np.array(n).reshape(-1,4) 
        self.a = a      
        self.n = n 
        self.i = i
        self.rindex = np.load(os.path.join(self.DIR, "RINDEX.npy"))  

    def __call__(self, name):
        a = self.a
        i = self.i
        jk = np.where( i == name )
        j = jk[0][0]  
        k = jk[1][0]
        q = a[:,j,k]
        return q 


class BetaInverseScan(L4CerenkovTest):
    def __init__(self):
        self.load(self.__class__.__name__)

        w = np.load(os.path.join(self.DIR, "SampleWavelengths_.npy"))
        self.w = w 

        a = self.a 
        n = self.n 

        jk = np.where( i == "materialIndex_branch" )
        j = jk[0][0]  
        k = jk[1][0]
        materialIndex_branch = a[:,j,k].copy().view(np.uint32)
        materialIndex = materialIndex_branch.view(np.uint32)[0::2]  
        branch = materialIndex_branch.view(np.uint32)[1::2]  
        assert np.all( materialIndex == 0 )

        self.materialIndex = materialIndex
        self.branch = branch 

    def __repr__(self):
        return repr(self.i)

    def plot(self):
        t = self
        charge = t('charge')
        beta = t('beta')
        Rfact = t('Rfact')
        BetaInverse = t('BetaInverse')

        Pmin = t('Pmin')
        Pmax = t('Pmax')
        nMin = t('nMin')
        nMax = t('nMax')

        CAImax = t('CAImax')
        materialIndex, branch = self.materialIndex, self.branch
        dp = t('dp')
        ge = t('ge')

        CAImin = t('CAImin')
        NumPhotons = t('NumPhotons')
        gamma = t('gamma')
        Padding0 = t('Padding0')

        fig, axs = plt.subplots(4, sharex=True)
        col = "rgb"
        for i in range(3):
            ax = axs[i]
            sel = np.where( branch == i )
            ax.plot( BetaInverse[sel], NumPhotons[sel], label="branch %d" % i, c=col[i])
            axs[3].plot( BetaInverse[sel], NumPhotons[sel], c=col[i])
            pass
               
        pass
        fig.legend() 
        fig.show()

        if 0:
            dom = np.arange(250,650,1) 
            h,_ = np.histogram(self.w, dom) 
            
            fig, ax = plt.subplots()
            ax.plot( dom[:-1], h )
            fig.show()





class SampleWavelengths(L4CerenkovTest):
    """
    ::

        ## for LS    nMin 1.4536 nMax 1.7930 
        ## so setting BetaInverse to 1.8 means the tail is never run
        ## making sin
       
        ./L4CerenkovTest.sh 1.8

        In [1]: np.all( ht[:,0] == 1 )
        Out[1]: True

        In [2]: np.all( ht[:,1] == 0 )
        Out[2]: True

    """
    def __init__(self):
        self.load(self.__class__.__name__)




if 0:
    bis = BetaInverseScan()  
    print(bis)
    bis.plot()

if 1:
    t = SampleWavelengths()

    rindex = t.rindex


    e = t("sampledEnergy")*1e6 
    w = t("sampledWavelength") 
    r = t("sampledRI")  
    c = t("cosTheta") 

    s2 = t("sin2Theta")
    bi = t("BetaInverse") 
    ht = t("head_tail").copy().view(np.uint32).reshape(-1,2)
    cc = t("continue_condition").copy().view(np.uint32).reshape(-1,2)

    h_e = np.histogram(e, 100)
    h_w = np.histogram(w, 100)
    h_r = np.histogram(r, 100)
    h_c = np.histogram(c, 100)
    h_s2 = np.histogram(s2, 100)

    nrows = 2
    ncols = 3 
    fig, axs = plt.subplots(nrows,ncols)



    ax = axs[0,0]
    ax.plot( h_e[1][:-1], h_e[0], label="e", drawstyle="steps" )
    ax.set_ylim( 0, h_e[0].max()*1.1 )
    ax.legend()

    ax = axs[0,1]
    ax.plot( h_w[1][:-1], h_w[0], label="w", drawstyle="steps" )
    ax.set_ylim( 0, h_w[0].max()*1.1 )
    ax.legend()

    ax = axs[0,2]
    ax.plot( h_r[1][:-1], h_r[0], label="r", drawstyle="steps" )
    ax.set_ylim( 0, h_r[0].max()*1.1 )
    ax.legend()

    ax = axs[1,0]
    ax.plot( h_c[1][:-1], h_c[0], label="c", drawstyle="steps" )
    ax.set_ylim( 0, h_c[0].max()*1.1 )
    ax.legend()

    ax = axs[1,1]
    ax.plot( h_s2[1][:-1], h_s2[0], label="s2", drawstyle="steps" )
    ax.set_ylim( 0, h_s2[0].max()*1.1 )
    ax.legend()

    ax = axs[1,2]
    ax.plot( rindex[:,0]*1e6, rindex[:,1], label="rindex", drawstyle="steps" )
    ax.set_ylim( 1, 2 )
    ax.legend()




    fig.show()














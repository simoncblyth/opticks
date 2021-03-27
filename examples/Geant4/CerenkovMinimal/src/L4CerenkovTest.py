#/usr/bin/env python

import os, numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
pass

np.set_printoptions(suppress=False, precision=8)   

class L4CerenkovTest(object):
    def __init__(self, dir_="/tmp"):
        a = np.load(os.path.join(dir_, "L4CerenkovTest.npy"))
        n = list(map(str.strip,open(os.path.join(dir_, "L4CerenkovTest.txt")).readlines()))
        i = np.array(n).reshape(4,4) 

        jk = np.where( i == "materialIndex_branch" )
        j = jk[0][0]  
        k = jk[1][0]
        materialIndex_branch = a[:,j,k].copy().view(np.uint32)

        materialIndex = materialIndex_branch.view(np.uint32)[0::2]  
        branch = materialIndex_branch.view(np.uint32)[1::2]  
        assert np.all( materialIndex == 0 )

        self.a = a      
        self.n = n 
        self.i = i 
        self.materialIndex = materialIndex
        self.branch = branch 

    def __call__(self, name):
        a = self.a
        if name in ['materialIndex', 'branch']:
            q = getattr(self, name)
        else:
            jk = np.where( self.i == name )
            j = jk[0][0]  
            k = jk[1][0]
            q = a[:,j,k]
        pass
        return q 

    def __repr__(self):
        return repr(self.i)


if __name__ == '__main__':
    t = L4CerenkovTest()  
    print(t)

    charge = t('charge')
    beta = t('beta')
    Rfact = t('Rfact')
    BetaInverse = t('BetaInverse')

    Pmin = t('Pmin')
    Pmax = t('Pmax')
    nMin = t('nMin')
    nMax = t('nMax')

    CAImax = t('CAImax')
    materialIndex, branch = t('materialIndex'), t('branch')
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



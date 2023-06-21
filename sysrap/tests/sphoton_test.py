#!/usr/bin/env python
import numpy as np
from opticks.ana.fold import Fold

import matplotlib.pyplot as plt
SIZE = np.array([1280,720])

if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    a = t.test_dot_pol_cross_mom_nrm     

    fig, ax = plt.subplots(1, figsize=SIZE/100.)

    ii = np.linspace(0,len(a),len(a))  
    ax.scatter(  a[:,0], a[:,1], label="a[:,0] [:,1]  fr, pot " )
    ax.scatter(  a[:,0], a[:,2], label="a[:,0] a[:,2] fr, pot/mct " )
    ax.legend()

    fig.show()

pass



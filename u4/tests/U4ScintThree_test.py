#!/usr/bin/env python
"""


In [3]: np.all(t.icdf[0] == t.lab_cmp_icdf )
Out[3]: np.True_

In [4]: np.all(t.icdf[1] == t.ppo_cmp_icdf )
Out[4]: np.True_

In [5]: np.all(t.icdf[2] == t.bis_cmp_icdf )
Out[5]: np.True_

In [6]: np.all( t.lab_cmp_icdf == t.bis_cmp_icdf )
Out[6]: np.True_


"""

import sys, numpy as np
from opticks.ana.fold import Fold

from opticks.ana.pvplt import *

MODE =  int(os.environ.get("MODE", "0"))
assert MODE in [0,2,-2,3,-3]

if __name__ == '__main__':
    t = Fold.Load(symbol="t")  # from $FOLD
    print(repr(t))

    hc_eVnm = 1239.84198
    nm = np.linspace(80,800,721)

    lab_abs = np.interp( nm, hc_eVnm/(1e6*t.lab_abs[:,0])[::-1], t.lab_abs[:,1][::-1] )
    ppo_abs = np.interp( nm, hc_eVnm/(1e6*t.ppo_abs[:,0])[::-1], t.ppo_abs[:,1][::-1] )
    bis_abs = np.interp( nm, hc_eVnm/(1e6*t.bis_abs[:,0])[::-1], t.bis_abs[:,1][::-1] )
    tot_abs = 1./lab_abs + 1./ppo_abs + 1./bis_abs

    p_lab_abs = (1./lab_abs)/tot_abs
    p_ppo_abs = (1./ppo_abs)/tot_abs
    p_bis_abs = (1./bis_abs)/tot_abs



    lab_rem = np.interp( nm, hc_eVnm/(1e6*t.lab_rem[:,0])[::-1], t.lab_rem[:,1][::-1] )
    ppo_rem = np.interp( nm, hc_eVnm/(1e6*t.ppo_rem[:,0])[::-1], t.ppo_rem[:,1][::-1] )
    bis_rem = np.interp( nm, hc_eVnm/(1e6*t.bis_rem[:,0])[::-1], t.bis_rem[:,1][::-1] )
    tot_rem = lab_rem + ppo_rem + bis_rem





    np.all( t.lab_cmp == t.bis_cmp )

    lab_icdf = t.lab_cmp_icdf[0].reshape(-1)
    ppo_icdf = t.ppo_cmp_icdf[0].reshape(-1)
    bis_icdf = t.bis_cmp_icdf[0].reshape(-1)
    prob = np.linspace(0,1,len(lab_icdf))

    if MODE == 2:
        if 0:
            label = "3 way probabilites from abslen"
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label)
            ax = axs[0]
            ax.set_xlim(200,600)
            ax.set_aspect('auto')
            ax.scatter( nm, p_lab_abs, s=1, c="r", label="lab" )
            ax.scatter( nm, p_ppo_abs, s=1, c="g", label="ppo" )
            ax.scatter( nm, p_bis_abs, s=1, c="b", label="bis" )
            ax.legend()
            fig.show()
        pass

        if 0:
            label = "icdf"
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label)
            ax = axs[0]
            ax.set_aspect('auto')
            #ax.set_ylim(.min(),icdf_0.max())
            ax.set_xlim(0,1)
            ax.scatter( prob, lab_icdf, s=1, c="r", label="lab_icdf" )
            ax.scatter( prob, ppo_icdf, s=1, c="g", label="ppo_icdf" )
            ax.scatter( prob, bis_icdf, s=1, c="b", label="bis_icdf" )
            ax.legend()
            fig.show()
        pass


        if 1:
            label = "rem"
            fig, axs = mpplt_plotter(nrows=1, ncols=3, label=label)

            qq = [lab_rem, ppo_rem, bis_rem]
            ll = ["lab_rem","ppo_rem","bis_rem"]
            cc = ["r","g","b" ]

            for i,ax in enumerate(axs):
                ax.set_aspect('auto')
                ax.scatter( nm, qq[i], s=1, c=cc[i], label=ll[i] )
                ax.legend()
            pass
            fig.show()
        pass

    pass

    rc = 0
    sys.exit(rc)



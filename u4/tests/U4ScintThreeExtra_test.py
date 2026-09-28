#!/usr/bin/env python
"""


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

    # convert energy domain into wavelength nm domain - swap order for ascending nm - interpolate onto fixed nm domain
    lab_abs = np.interp( nm, hc_eVnm/(1e6*t.lab_abs[:,0])[::-1], t.lab_abs[:,1][::-1] )
    ppo_abs = np.interp( nm, hc_eVnm/(1e6*t.ppo_abs[:,0])[::-1], t.ppo_abs[:,1][::-1] )
    bis_abs = np.interp( nm, hc_eVnm/(1e6*t.bis_abs[:,0])[::-1], t.bis_abs[:,1][::-1] )

    # invert absorption lengths to give absorption coefficient
    lab_coe = 1./lab_abs
    ppo_coe = 1./ppo_abs
    bis_coe = 1./bis_abs
    ## tot_abs = 1./lab_abs + 1./ppo_abs + 1./bis_abs   - former confusing naming

    # add the three to give total abs coefficient
    tot_coe = lab_coe + ppo_coe + bis_coe

    # invert to give total absorption length
    tot_abs = 1./tot_coe


    # probabilities or fractions of the three species
    p_lab = lab_coe/tot_coe
    p_ppo = ppo_coe/tot_coe
    p_bis = bis_coe/tot_coe

    # planned float4 tex
    float4_tex = np.c_[tot_abs, p_lab, p_ppo, p_bis]

    ## TODO: reproduce the above in U4ScintThree.h - probably with NP.hh additions - and compare with the above


    # add probabilities - should get very close to 1.
    p_tot = p_lab + p_ppo + p_bis
    p_tot_delta = np.max(np.abs(p_tot - 1.))
    assert p_tot_delta < 1e-10


    lab_rem = np.interp( nm, hc_eVnm/(1e6*t.lab_rem[:,0])[::-1], t.lab_rem[:,1][::-1] )
    ppo_rem = np.interp( nm, hc_eVnm/(1e6*t.ppo_rem[:,0])[::-1], t.ppo_rem[:,1][::-1] )
    bis_rem = np.interp( nm, hc_eVnm/(1e6*t.bis_rem[:,0])[::-1], t.bis_rem[:,1][::-1] )
    tot_rem = lab_rem + ppo_rem + bis_rem






    if MODE == 2:
        if 0:
            label = "3 way probabilites from abslen"
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=label)
            ax = axs[0]
            ax.set_xlim(200,600)
            ax.set_aspect('auto')
            ax.scatter( nm, p_lab, s=1, c="r", label="lab" )
            ax.scatter( nm, p_ppo, s=1, c="g", label="ppo" )
            ax.scatter( nm, p_bis, s=1, c="b", label="bis" )
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



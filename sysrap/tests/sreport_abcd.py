#!/usr/bin/env python

import os, numpy as np
from opticks.ana.fold import Fold
from opticks.ana.npmeta import NPMeta
from opticks.sysrap.tests.sreport import Substamp 
from opticks.sysrap.tests.sreport import RUN_META 

MODE = 2
PLOT =  os.environ.get("PLOT","?plot") 
COMMANDLINE = os.environ.get("COMMANDLINE", "")


if MODE != 0:
    from opticks.ana.pvplt import * 
pass


class ABCD_Substamp_ALL_Etime_vs_Photon(object):
    def __init__(self, *rr):
        title = RUN_META.ABCD_Title(*rr)

        ratt = "abcd"
        assert len(rr) <= len(ratt) 
        _qn0 = None 
        for i,r in enumerate(rr):
            qf = "%sf" % ratt[i]
            qt = "%st" % ratt[i]
            qn = "%sn" % ratt[i]
            qh = "%sh" % ratt[i]

            _qf = r.substamp.a
            _qt = Substamp.ETime(_qf)
            _qn = r.submeta_NumPhotonCollected.a
            _qh = Substamp.Subcount(_qf, "hit")

            setattr(self, qf, _qf )
            setattr(self, qt, _qt )  
            setattr(self, qn, _qn ) 
            setattr(self, qh, _qh ) 

            if i == 0:
                _qn0 = _qn
            else:
                assert np.all( _qn0 == _qn )
            pass
        pass
        photon = _qn0[:,0]/1e6


        fontsize = 20
        YSCALE_ = ["log", "linear" ] 
        YSCALE = os.environ.get("YSCALE", YSCALE_[1])
        assert YSCALE in YSCALE_
 

 
        YMIN = float(os.environ.get("YMIN", "1e-2"))
        YMAX = float(os.environ.get("YMAX", "45"))

        if MODE == 2:
            fig, axs = mpplt_plotter(nrows=1, ncols=1, label=title, equal=False)
            ax = axs[0]

            #sli = slice(3,12)
            #sli = slice(None)

            deg = 1  # 1:lin 2:par 
        

            for i,r in enumerate(rr):
                qt = "%st" % ratt[i]
                _qt = getattr(self, qt) 


                # from A_SLI, B_SLI, if present otherwise from SLI, default "3:12"

                Q_SLI = "%s_SLI" % ratt[i].upper() 
                if Q_SLI in os.environ:
                    _SLI = os.environ.get(Q_SLI)
                else:
                    SLI_DEF="3:12"
                    _SLI = os.environ.get("SLI", SLI_DEF)
                pass 
                SLI = list(map(int, _SLI.split(":")))
                sli = slice(*SLI)



                q_fit = "%s_fit" % ratt[i]
                r_fit = np.poly1d(np.polyfit(photon[sli], _qt[sli], deg))
                r_fit_label = "%s : linefit( slope %10.3f  intercept %10.3f )" % (r.symbol, r_fit.coef[0], r_fit.coef[1])
                
                setattr(self, q_fit, r_fit ) 

                r_GPU_label = RUN_META.GPUMeta(r)
                r_RNG_label = RUN_META.QSim__RNGLabel(r)
                r_label = "%s (%s)" % (r_GPU_label, r_RNG_label) 

                ax.scatter( photon, _qt, label=r_label )
                ax.plot( photon[sli], r_fit(photon[sli]), label=r_fit_label )
            pass
            ax.set_xlim( -5, 105 ); 
            ax.set_ylim( YMIN, YMAX );  # 50*200 = 1e4
            ax.set_yscale(YSCALE)
            ax.set_ylabel("Event time (seconds)", fontsize=fontsize )
            ax.set_xlabel("Number of Photons (Millions)", fontsize=fontsize )
            ax.legend()
            ax.legend()
            fig.show()
        pass





if __name__ == '__main__':

    print("[sreport_abcd.py:PLOT[%s]" % PLOT ) 

    REPS="ABCD"
    rr = []
    for i, Q in enumerate(REPS): 
       
        q = os.environ.get(Q, None)
        if q is None: 
            continue
        pass
        QRF = "$%s_SREPORT_FOLD" % Q

        print("[sreport_abcd.py:r = Fold.Load(%s, symbol=%s)" % (QRF, q) ) 
        r = Fold.Load(QRF, symbol=q )
        print("]sreport_abcd.py:r = Fold.Load(%s, symbol=%s)" % (QRF, q) ) 

        print("[sreport_abcd.py:repr(r)" ) 
        print(repr(r))
        print("]sreport_abcd.py:repr(r)" )
        rr.append(r) 
    pass
    print("]sreport_ab.py:PLOT[%s]" % PLOT ) 

    if PLOT.startswith("ABCD_Substamp_ALL_Etime_vs_Photon"): 
        abcd = ABCD_Substamp_ALL_Etime_vs_Photon(*rr)
    pass






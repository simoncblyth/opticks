#!/usr/bin/env python

import numpy as np, textwrap
from opticks.ana.fold import Fold
import matplotlib.pyplot as plt
SIZE = np.array([1280,720])

def Vacuum_kludge(ff, names=["mat","oldmat"]):
    for f in ff:
        print("Vacuum_kludge %s " % f.base)
        for m in names:
            q = getattr(f, m, None)
            if q is None: continue
            if np.all( q[16,0,:,1] == 1e9 ):
                print("%s : Vacuum 1e9 kludge reduce to 1e6 " % m )
                q[16,0,:,1] = 1e6 
            else:
                print("%s : Not doing Vacuum kludge " % m )
            pass
        pass
    pass


def eprint(exprs):
    print("eprint(\"\"\"%s\"\"\")" % exprs  )
    for expr in list(filter(None,textwrap.dedent(exprs).split("\n"))):
        print(expr)
        if expr[0] in " #": continue
        print(eval(expr))
        print("\n")        
    pass



if __name__ == '__main__':
    t = Fold.Load(symbol="t")
    print(repr(t))
    Vacuum_kludge([t])
    ab = np.abs( t.mat - t.oldmat ) 

    obn = np.array(t.oldbnd_names)
    oop = t.oldoptical.reshape(-1,4,4).view(np.int32)  
    assert len(obn) == len(oop)   

    bn = np.array(t.bnd_names)
    op = t.optical
    assert len(bn) == len(op)
    


    eprint("""
    np.all( np.array( t.mat_names) == np.array( t.oldmat_names ))  
    t.mat.shape == t.oldmat.shape
    np.unique(np.where( np.abs(t.mat - t.oldmat) > 1e-3 )[0])
    np.array(t.mat_names)[np.unique(np.where( np.abs(t.mat - t.oldmat) > 1e-3 )[0])] 

    #  RINDEX     ABSLENGTH  RAYLEIGH   REEMISSIONPROB   GROUPVEL 
    np.max(ab, axis=2).reshape(-1,8)   # max deviation across wavelength domain 
    np.c_[np.arange(len(t.mat_names)),np.array(t.mat_names)] 
    """
    )


if 0:
    BASE = "$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim"

    ## t.mat[np.where( t.mat == 300. )] = 299.792458  # GROUPVEL default kludge 

    rayleigh_Water_idx = np.where( np.array(t.rayleigh_names)  == "Water" )[0][0] 
    rayleigh_Water_ = t.rayleigh[rayleigh_Water_idx]
    rayleigh_Water_ni = rayleigh_Water_[-1,-1].view(np.int64)
    rayleigh_Water = rayleigh_Water_[:rayleigh_Water_ni]

    #wl = np.linspace(60.,820.,761)
    wl = t.wavelength 
    en = t.energy 

    doms = {"wl":wl, "en":en }
    DOM = os.environ.get("DOM", "en")
    dom = doms[DOM[:2]]

    if DOM == "ensel4":
        sel = np.where( dom < 4 )
    else:
        sel = slice(None)
    pass
  
    plot = "RINDEX"
    #plot = "GROUPVEL"
    #plot = "RAYLEIGH"
    #plot = "ABSLENGTH"

    PLOT = os.environ.get("PLOT", plot)
    qwns="mat oldmat".split()

    MM = [4,11,14,17,18,19]

    prop = {}
    prop["RINDEX"] = (0,0)
    prop["ABSLENGTH"] = (0,1)
    prop["RAYLEIGH"] = (0,2)
    prop["REEMISSIONPROB"] = (0,3)
    prop["GROUPVEL"] = (1,0)

    for M in MM:
        MAT = t.mat_names[M]

        title = "PLOT:%s DOM:%s M:%d MAT:%s " % (PLOT,DOM, M, MAT) 
        print(title)

        fig, ax = plt.subplots(1, figsize=SIZE/100.)
        fig.suptitle(title)

        if PLOT.split("_")[0] == "DIFF":
            a = getattr(t, qwns[0], None)
            b = getattr(t, qwns[1], None)
            p = prop.get(PLOT.split("_")[1], None)
            aq = a[M,p[0],:,p[1]]
            bq = b[M,p[0],:,p[1]]
            abq = aq - bq

            label = "%s %s" % (PLOT, "%s-%s" % (qwns[0],qwns[1]) )            
            ax.plot( dom[sel], abq[sel] , label=label )

            if PLOT == "DIFF_RAYLEIGH" and DOM[:2] == "en" and MAT == "Water":

                rayleigh_Water_en = rayleigh_Water[:,0]*1e6
                if DOM == "ensel4":
                    rayleigh_Water_ensel = np.where(rayleigh_Water_en < 4 )
                else:
                    rayleigh_Water_ensel = slice(None)
                pass
                for xc in rayleigh_Water_en[rayleigh_Water_ensel]:
                    ax.axvline(x=xc)
                pass
            pass
        else:        
            for qwn in qwns:
                a = getattr(t, qwn, None)
                assert not a is None
                p = prop.get(PLOT, None)
                assert not p is None
                aq = a[M,p[0],:,p[1]]
                label = "%s %s" % (PLOT, qwn)            

                ax.plot( dom[sel], aq[sel], label=label )

                REL = "stree/material/%(MAT)s/%(PLOT)s.npy" % locals()
                path = os.path.expandvars(os.path.join(BASE,REL))  

                if DOM[:2] == "en" and os.path.exists(path):
                    meas = np.load(path)
                    meas_en = meas[:,0]*1e6
                    meas_va = meas[:,1]
                    if DOM == "ensel4":
                        meas_sel = np.where(meas_en < 4 )
                    else:
                        meas_sel = slice(None)
                    pass
                    ax.scatter( meas_en[meas_sel], meas_va[meas_sel], label="meas" )
                pass

                RID = "stree/material/%(MAT)s/RINDEX.npy" % locals()
                ri_path = os.path.expandvars(os.path.join(BASE,RID))  

                if DOM[:2] == "en" and os.path.exists(ri_path):
                    ri = np.load(ri_path)
                    ri_en = ri[:,0]*1e6
                    ri_va = ri[:,1] 
                    if DOM == "ensel4":
                        ri_ensel = np.where(ri_en < 4)
                    else:
                        ri_ensel = slice(None)
                    pass
                    for xc in ri_en[ri_ensel]:
                        ax.axvline(x=xc)
                    pass
                pass
                #if PLOT == "RAYLEIGH" and DOM[:2] == "en" and MAT == "Water":
                #    ax.scatter( rayleigh_Water[:,0]*1e6, rayleigh_Water[:,1], label="rayleigh_Water") 
                #pass
            pass
        pass
        ax.legend()
        fig.show()
    pass

if 0:
    o = Fold.Load("/tmp/SBnd_test", symbol="o")
    print(repr(o))

    s = Fold.Load("/tmp/blyth/opticks/U4TreeCreateTest/stree", symbol="s")
    print(repr(s))

    Vacuum_kludge([t,o,s])


    exprs = """
    np.all( t.old_optical == o.optical )
    np.all( np.array( t.mat_names ) == np.array( o.mat_names ) ) 
    """

    for expr in list(filter(None,textwrap.dedent(exprs).split("\n"))):
        print(expr)
        print(eval(expr))
    pass

    assert len(t.mat) == len(o.mat)

    print("ij")
    for i in range(len(o.mat)): 
        for j in range(2):
            expr= "np.all( o.mat[%(i)d,%(j)d] == s.mat[%(i)d,%(j)d] )" % locals()
            print(" %s : %s " % (expr, eval(expr)))
        pass
    pass

    print("ijk")
    for i in range(len(o.mat)):
        tname = t.mat_names[i]
        oname = o.mat_names[i]
        assert( tname == oname )
        print( "\n i : %(i)d  %(tname)s " % locals() )
        for j in range(2):
            print( " j : %(j)d " % locals() )
            for k in range(4):
                expr= "len(np.where( np.abs( o.mat[%(i)d,%(j)d,:,%(k)d] - s.mat[%(i)d,%(j)d,:,%(k)d] ) > 1e-4)[0])" % locals()
                print(" %s : %s " % (expr, eval(expr)))
            pass
        pass
    pass
pass
 

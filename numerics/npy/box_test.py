#!/usr/bin/env python
"""
BoxInBox Opticks vs cfg4
==================================

Without and with cfg4 runs::

   ggv-;ggv-box-test 
   ggv-;ggv-box-test --cfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-box-test --cfg4 --load


::

       0.000 100.004  0.129 100.002
                      1:BoxInBox   -1:BoxInBox           c2 
                  8d       432190       431510             0.54  [2 ] TO SA
                  4d        56135        56764             3.50  [2 ] TO AB
                 86d        10836        10828             0.00  [3 ] TO SC SA
                 46d          695          723             0.55  [3 ] TO SC AB
                866d          140          161             1.47  [4 ] TO SC SC SA
                466d            3           10             0.00  [4 ] TO SC SC AB
               8666d            0            3             0.00  [5 ] TO SC SC SC SA
               4666d            0            1             0.00  [5 ] TO SC SC SC AB
                  3d            1            0             0.00  [2 ] TO MI
                          500000       500000         1.21 
                      1:BoxInBox   -1:BoxInBox           c2 
                  44       488325       488274             0.00  [2 ] MO MO
                 444        11531        11551             0.02  [3 ] MO MO MO
                4444          143          171             2.50  [4 ] MO MO MO MO
               44444            0            4             0.00  [5 ] MO MO MO MO MO
                   4            1            0             0.00  [1 ] MO
                          500000       500000         0.84 


Issues
-------

Refractive index of MO mismatch, or somehow accessing wrong material prop::

    INFO:env.numerics.npy.evt:Evt seqs ['TO SA'] 
    sa(Op)
      0 z    300.000    300.000    300.000 r      0.000    100.004     50.002  t      0.098      0.098      0.098    smry m1/m2   4/ 12 MO/Rk  124 (123)  13:TO  
      1 z   -300.000   -300.000   -300.000 r      0.000    100.004     50.002  t      3.070      3.070      3.070    smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    sb(G4)
      0 z    300.000    300.000    300.000 r      0.129    100.002     50.066  t      0.098      0.098      0.098    smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z   -300.000   -300.000   -300.000 r      0.129    100.002     50.066  t      3.265      3.265      3.265    smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  



"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
from env.numerics.npy.evt import Evt

X,Y,Z,W = 0,1,2,3


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    np.set_printoptions(precision=4, linewidth=200)

    plt.ion()
    plt.close()

    tag = "1"
    a = Evt(tag="%s" % tag, src="torch", det="BoxInBox")
    b = Evt(tag="-%s" % tag , src="torch", det="BoxInBox")

    a0 = a.rpost_(0)
    a0r = np.linalg.norm(a0[:,:2],2,1)

    b0 = b.rpost_(0)
    b0r = np.linalg.norm(b0[:,:2],2,1)

    print " ".join(map(lambda _:"%6.3f" % _, (a0r.min(),a0r.max(),b0r.min(),b0r.max())))

    hcf = a.history.table.compare(b.history.table)
    print hcf

    mcf = a.material.table.compare(b.material.table)
    print mcf


    np.set_printoptions(formatter={'int':hex})
    ahm = np.dstack([a.seqhis,a.seqmat])[0]
    bhm = np.dstack([b.seqhis,b.seqmat])[0]

    print "ahm (Op)\n", ahm
    print "bhm (G4)\n", bhm



    sa = Evt(tag="%s" % tag, src="torch", det="BoxInBox", seqs=["TO SA"])
    sb = Evt(tag="-%s" % tag, src="torch", det="BoxInBox", seqs=["TO SA"])

    print "sa(Op)"
    sa_zrt = sa.zrt_profile(2)
  
    print "sb(G4)"
    sb_zrt = sb.zrt_profile(2)





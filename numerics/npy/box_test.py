#!/usr/bin/env python
"""
BoxInBox Opticks vs cfg4
==================================

Without and with cfg4 runs::

   ggv-;ggv-box-test 
   ggv-;ggv-box-test --cfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-box-test --cfg4 --load


Issues
-------

Huh, suspiciously good BoxInBox match for MineralOil whereas the pmt_test would 
suggest this should be discrepant.

::

    In [12]: a.history_table()
    Evt(1,"torch","BoxInBox","", seqs="[]")
                              noname 
                      8d       432190       [2 ] TO SA
                      4d        56135       [2 ] TO AB
                     86d        10836       [3 ] TO SC SA
                     46d          695       [3 ] TO SC AB
                    866d          140       [4 ] TO SC SC SA
                    466d            3       [4 ] TO SC SC AB
                      3d            1       [2 ] TO MI
                              500000 

    In [13]: b.history_table()
    Evt(-1,"torch","BoxInBox","", seqs="[]")
                              noname 
                      8d       431393       [2 ] TO SA
                      4d        56862       [2 ] TO AB
                     86d        10841       [3 ] TO SC SA
                     46d          728       [3 ] TO SC AB
                    866d          160       [4 ] TO SC SC SA
                    466d           12       [4 ] TO SC SC AB
                   8666d            3       [5 ] TO SC SC SC SA
                   4666d            1       [5 ] TO SC SC SC AB
                              500000 



Bulk absorption depends on ABSLENGTH of the material::

    simon:cfg4 blyth$ ggv --mat 3
    /Users/blyth/env/bin/ggv.sh dumping cmdline arguments
    --mat
    3
    [2016-02-26 13:32:52.162150] [0x000007fff7057a31] [info]    Opticks::preconfigure mode Interop detector dayabay
    [2016-Feb-26 13:32:52.166179]:info: GMaterialLib::dump NumMaterials 38
    [2016-Feb-26 13:32:52.166767]:info: GPropertyMap<T>::  3       material m:MineralOil k:refractive_index absorption_length scattering_length reemission_prob MineralOil
                  domain    refractive_index   absorption_length   scattering_length     reemission_prob
                      60               1.434                11.1                 850                   0
                      80               1.434                11.1                 850                   0
                     100               1.434                11.1                 850                   0
                     120               1.434                11.1                 850                   0
                     140             1.64207                11.1                 850                   0
                     160             1.75844                11.1                 850                   0
                     180             1.50693                11.1                 850                   0
                     200             1.59558             10.7949             851.716                   0
                     220             1.57716             10.6971             2201.56                   0
                     240             1.55875                11.5             3551.41                   0
                     260             1.54033             11.3937             4901.25                   0
                     280             1.52192                10.9              6251.1                   0
                     300              1.5035             39.6393             7602.84                   0
                     320             1.49829             117.679               11675                   0
                     340             1.49307             490.025             15747.2                   0
                     360             1.48786              1078.9             19819.4                   0
                     380             1.48264             4941.76             23891.6                   0
                     400             1.47743             11655.2             27963.7                   0
                     420             1.47458             24706.1             36028.8                   0
                     440             1.47251             25254.7             45367.7                   0
                     460             1.47063             24925.3               52039                   0
                     480             1.46876               24277             58710.2                   0
                     500             1.46734             23628.8               68425                   0
                     520              1.4661             22980.5             81100.8                   0
                     540             1.46487             22332.2             93776.7                   0
                     560             1.46369             21277.4              117807                   0
                     580             1.46252             18523.2              152790                   0
                     600             1.46158             14966.4              181999                   0
                     620             1.46081             7061.42              205618                   0
                     640             1.46004             4159.07              229236                   0
                     660             1.45928             5311.87              252855                   0
                     680             1.45851             5615.17              276473                   0
                     700             1.45796             4603.84              300155                   0
                     720             1.45764             3697.27              340165                   0
                     740             1.45733             1365.95              380175                   0
                     760             1.45702              837.71              420184                   0
                     780             1.45671             2274.95              460194                   0
                     800              1.4564             2672.76              500000                   0
                     820              1.4564             1614.62              500000                   0


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





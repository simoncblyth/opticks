#!/usr/bin/env python
"""
box_test.py : BoxInBox Opticks vs Geant4 comparisons
=======================================================

Without and with cfg4 runs::

   ggv-;ggv-box-test 
   ggv-;ggv-box-test --tcfg4 

Visualize the cfg4 created evt in interop mode viewer::

   ggv-;ggv-box-test --tcfg4 --load


.. code-block:: py 

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

    INFO:opticks.ana.evt:Evt seqs ['TO SA'] 
    sa(Op)
      0 z    300.000    300.000    300.000 r      0.000    100.004     50.002  t      0.098      0.098      0.098    smry m1/m2   4/ 12 MO/Rk  124 (123)  13:TO  
      1 z   -300.000   -300.000   -300.000 r      0.000    100.004     50.002  t      3.070      3.070      3.070    smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    sb(G4)
      0 z    300.000    300.000    300.000 r      0.129    100.002     50.066  t      0.098      0.098      0.098    smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z   -300.000   -300.000   -300.000 r      0.129    100.002     50.066  t      3.265      3.265      3.265    smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  


Running this in debugger::

   ggv-;ggv-box-test --cfg4 --dbg

Shows that steps exiting the world are not well handled by G4OpBoundaryProcess, as 
a step status of fWorldBoundary results in cop out with NotAtBoundary status::


     183         G4bool isOnBoundary =
     184                 (pStep->GetPostStepPoint()->GetStepStatus() == fGeomBoundary);
     185 
     186         if (isOnBoundary) {
     187            Material1 = pStep->GetPreStepPoint()->GetMaterial();
     188            Material2 = pStep->GetPostStepPoint()->GetMaterial();
     189         } else {
     190            theStatus = NotAtBoundary;
     191            if ( verboseLevel > 0) BoundaryProcessVerbose();
     192            return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     193         }


So add a Pyrex cube to the geometry::

     182     local test_config=(
     183                  mode=BoxInBox
     184                  analytic=1
     185                  shape=box
     186                  boundary=Rock//perfectAbsorbSurface/MineralOil
     187                  parameters=0,0,0,300
     188                  
     189                  shape=box
     190                  boundary=MineralOil///Pyrex
     191                  parameters=0,0,0,100
     192                    )



Checking refractive index with pmt-test gives the expected value by comparison with ggv --mat 3::

    ggv-;ggv-pmt-test --cfg4 --dbg
   

    (lldb) p thePhotonMomentum*1e6
    (double) $6 = 3.2627417774210459

    (lldb) p Rindex1
    (G4double) $7 = 1.4826403856277466


::

    simon:env blyth$ ggv --mat  3
    /Users/blyth/env/bin/ggv.sh dumping cmdline arguments
    --mat
    3
    [2016-03-04 13:14:14.853950] [0x000007fff7057a31] [info]    Opticks::preconfigure argv[0] /usr/local/env/optix/ggeo/bin/GMaterialLibTest mode Interop detector dayabay
    [2016-Mar-04 13:14:14.858301]:info: GMaterialLib::dump NumMaterials 38
    [2016-Mar-04 13:14:14.858686]:info: GPropertyMap<T>::  3       material m:MineralOil k:refractive_index absorption_length scattering_length reemission_prob MineralOil
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


Timing still off despite a quick check suggesting the refractive indices are matching::

    INFO:opticks.ana.evt:Evt seqs ['TO BT BT SA'] 
    sa(Op)
      0 z    300.000    300.000    300.000 r      0.000    100.004     50.002  t      0.098      0.098      0.098    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z     99.997     99.997     99.997 r      0.000    100.004     50.002  t      1.086      1.086      1.086    smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z    -99.997    -99.997    -99.997 r      0.000    100.004     50.002  t      2.063      2.063      2.063    smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      3 z   -300.000   -300.000   -300.000 r      0.000    100.004     50.002  t      3.052      3.052      3.052    smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    sb(G4)
      0 z    300.000    300.000    300.000 r      0.208    100.003     50.105  t      0.098      0.098      0.098    smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z     99.997     99.997     99.997 r      0.208    100.003     50.105  t      1.154      1.154      1.154    smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z    -99.997    -99.997    -99.997 r      0.208    100.003     50.105  t      2.130      2.130      2.130    smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      3 z   -300.000   -300.000   -300.000 r      0.208    100.003     50.105  t      3.180      3.180      3.180    smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  

Ahha, default time domain of 0:200 ns leads to excess imprecision over 0:5, so set timemax to 10::

    INFO:opticks.ana.evt:Evt seqs ['TO BT BT SA'] 
    sa(Op)
      0 z    300.000    300.000    300.000 r      0.000    100.004     50.002  t      0.100      0.100      0.100    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z     99.997     99.997     99.997 r      0.000    100.004     50.002  t      1.089      1.089      1.089    smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z    -99.997    -99.997    -99.997 r      0.000    100.004     50.002  t      2.062      2.062      2.062    smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      3 z   -300.000   -300.000   -300.000 r      0.000    100.004     50.002  t      3.051      3.051      3.051    smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    sb(G4)
      0 z    300.000    300.000    300.000 r      0.208    100.003     50.105  t      0.100      0.100      0.100    smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z     99.997     99.997     99.997 r      0.208    100.003     50.105  t      1.155      1.155      1.155    smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z    -99.997    -99.997    -99.997 r      0.208    100.003     50.105  t      2.128      2.128      2.128    smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      3 z   -300.000   -300.000   -300.000 r      0.208    100.003     50.105  t      3.183      3.183      3.183    smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  

    In [3]: 1.155 - 1.089
    Out[3]: 0.06600000000000006

    In [4]: 2.128 - 2.062
    Out[4]: 0.06600000000000028

    In [5]: 3.183 - 3.051
    Out[5]: 0.13199999999999967

    In [6]: 0.066*2
    Out[6]: 0.132


Chased where times and velocities are set in G4 in g4op- conclusion
is that a sneaky GROUPVEL property is added to the MaterialPropertyTable
for materials with RINDEX.  This GROUPVEL is used for optical photon
velocity and resulting times.  Subverting this by imposing a GROUPVEL which is
actually the phase velocity (c_light/RINDEX) brings Opticks and CFG4 timings
into alignment.

Running with CPropLib::m_groupvel_kludge = true gets times to agree with Opticks::

    INFO:opticks.ana.evt:Evt seqs ['TO BT BT SA'] 
    sa(Op)
      0 z    300.000    300.000    300.000 r      0.000    100.004     50.002  t      0.100      0.100      0.100    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z     99.997     99.997     99.997 r      0.000    100.004     50.002  t      1.089      1.089      1.089    smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z    -99.997    -99.997    -99.997 r      0.000    100.004     50.002  t      2.062      2.062      2.062    smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      3 z   -300.000   -300.000   -300.000 r      0.000    100.004     50.002  t      3.051      3.051      3.051    smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    sb(G4)
      0 z    300.000    300.000    300.000 r      0.208    100.003     50.105  t      0.100      0.100      0.100    smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z     99.997     99.997     99.997 r      0.208    100.003     50.105  t      1.089      1.089      1.089    smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z    -99.997    -99.997    -99.997 r      0.208    100.003     50.105  t      2.062      2.062      2.062    smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      3 z   -300.000   -300.000   -300.000 r      0.208    100.003     50.105  t      3.051      3.051      3.051    smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  


TODO: examine the GROUPVEL calc and see how best to do in Opticks ?


"""
import os, logging, numpy as np
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from opticks.ana.base import opticks_environment
from opticks.ana.evt import Evt

X,Y,Z,W = 0,1,2,3


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    opticks_environment()

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


    sel = "TO BT BT SA" 
    nst = len(sel.split())

    sa = Evt(tag="%s" % tag, src="torch", det="BoxInBox", seqs=[sel])
    sb = Evt(tag="-%s" % tag, src="torch", det="BoxInBox", seqs=[sel])

    print "sa(Op)"
    sa_zrt = sa.zrt_profile(nst)
  
    print "sb(G4)"
    sb_zrt = sb.zrt_profile(nst)





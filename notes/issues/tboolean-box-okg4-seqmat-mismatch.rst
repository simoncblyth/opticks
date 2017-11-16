tboolean-box-okg4-seqmat-mismatch
====================================


FIXED 
------

* somewhat unsatisfying fix, was to add yet another special case to CRecorder::posttrackWriteSteps


ISSUE : wrong 1st material for a first step BR eg "TO BR SA" 
----------------------------------------------------------------


::

     tboolean-;tboolean-box --okg4
     tboolean-;tboolean-box-p

     tboolean-;tboolean-box --okg4 --load --vizg4


APPROACH
----------

* made inroads into CRecorder monolith, breaking into CWriter, CDebug, CPhoton, CG4Ctx, ...


::

    tboolean-box--(){ cat << EOP 
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.polyconfig import PolyConfig
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="$TMP/$FUNCNAME")

    emitconfig = "photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1" 

    CSG.kwa = dict(poly="IM",resolution="20", verbosity="0",ctrl="0", containerscale="3", emitconfig=emitconfig  )

    container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container="1" )  # no param, container="1" switches on auto-sizing

    #box = CSG("box3", param=[300,300,200,0], emit=-1,  boundary="Vacuum//perfectSpecularSurface/GlassSchottF2" )  <-- infinite bounce : trancation behaviour difference
    box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2" )

    CSG.Serialize([container, box], args.csgpath )
    EOP
    }



    2017-11-16 13:49:19,212] p15095 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171116-1349 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy 
    B tboolean-box/torch/ -1 :  20171116-1349 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.72/5 =  0.14  (pval:0.982 prob:0.018)  
    0000               8d     88742     88743             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO SA
    0001             8ccd      9826      9830             0.00        1.000 +- 0.010        1.000 +- 0.010  [4 ] TO BT BT SA
    0002              8bd       684       684             0.00        1.000 +- 0.038        1.000 +- 0.038  [3 ] TO BR SA
    0003            8cbcd       618       613             0.02        1.008 +- 0.041        0.992 +- 0.040  [5 ] TO BT BR BT SA
    0004              86d        81        77             0.10        1.052 +- 0.117        0.951 +- 0.108  [3 ] TO SC SA
    0005           8cbbcd        27        33             0.60        0.818 +- 0.157        1.222 +- 0.213  [6 ] TO BT BR BR BT SA
    0006               4d         6        11             0.00        0.545 +- 0.223        1.833 +- 0.553  [2 ] TO AB
    0007            86ccd         6         1             0.00        6.000 +- 2.449        0.167 +- 0.167  [5 ] TO BT BT SC SA
    0008          8cbbbcd         4         2             0.00        2.000 +- 1.000        0.500 +- 0.354  [7 ] TO BT BR BR BR BT SA
    0009              4cd         3         1             0.00        3.000 +- 1.732        0.333 +- 0.333  [3 ] TO BT AB
    0010            8cc6d         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [5 ] TO SC BT BT SA
    0011             8b6d         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO SC BR SA
    0012           8cbc6d         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT BR BT SA
    0013            8c6cd         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT SC BT SA
    0014       bbbbbbb6cd         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] TO BT SC BR BR BR BR BR BR BR
    .                             100000    100000         0.72/5 =  0.14  (pval:0.982 prob:0.018)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.10/4 =  0.03  (pval:0.999 prob:0.001)  
    0000             1080     88742     88743             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO|SA
    0001             1880      9826      9830             0.00        1.000 +- 0.010        1.000 +- 0.010  [3 ] TO|BT|SA
    0002             1480       684       684             0.00        1.000 +- 0.038        1.000 +- 0.038  [3 ] TO|BR|SA
    0003             1c80       649       648             0.00        1.002 +- 0.039        0.998 +- 0.039  [4 ] TO|BT|BR|SA
    0004             10a0        81        77             0.10        1.052 +- 0.117        0.951 +- 0.108  [3 ] TO|SA|SC
    0005             1008         6        11             0.00        0.545 +- 0.223        1.833 +- 0.553  [2 ] TO|AB
    0006             18a0         8         3             0.00        2.667 +- 0.943        0.375 +- 0.217  [4 ] TO|BT|SA|SC
    0007             1808         3         1             0.00        3.000 +- 1.732        0.333 +- 0.333  [3 ] TO|BT|AB
    0008             14a0         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BR|SA|SC
    0009             1c20         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [4 ] TO|BT|BR|SC
    0010             1ca0         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|BR|SA|SC
    .                             100000    100000         0.10/4 =  0.03  (pval:0.999 prob:0.001)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000      1246.78/5 = 249.36  (pval:0.000 prob:1.000)  
    0000               12     88742     88743             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] Vm Rk
    0001             1232      9826      9830             0.00        1.000 +- 0.010        1.000 +- 0.010  [4 ] Vm F2 Vm Rk
    0002              122       765        77           562.17        9.935 +- 0.359        0.101 +- 0.011  [3 ] Vm Vm Rk
    0003              123         0       684           684.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] F2 Vm Rk
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
    0004            12332       618       614             0.01        1.007 +- 0.040        0.994 +- 0.040  [5 ] Vm F2 F2 Vm Rk
    0005           123332        27        33             0.60        0.818 +- 0.157        1.222 +- 0.213  [6 ] Vm F2 F2 F2 Vm Rk
    0006               22         6        11             0.00        0.545 +- 0.223        1.833 +- 0.553  [2 ] Vm Vm
    0007            12232         6         1             0.00        6.000 +- 2.449        0.167 +- 0.167  [5 ] Vm F2 Vm Vm Rk
    0008          1233332         4         2             0.00        2.000 +- 1.000        0.500 +- 0.354  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0009              332         3         1             0.00        3.000 +- 1.732        0.333 +- 0.333  [3 ] Vm F2 F2
    0010            12322         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [5 ] Vm Vm F2 Vm Rk
    0011       3333333332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0012           123322         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm Vm F2 F2 Vm Rk
    0013             1222         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Vm Vm Vm Rk
    .                             100000    100000      1246.78/5 = 249.36  (pval:0.000 prob:1.000)  


After the fix::

    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         0.62/4 =  0.16  (pval:0.960 prob:0.040)  
    0000               12     88742     88743             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] Vm Rk
    0001             1232      9826      9830             0.00        1.000 +- 0.010        1.000 +- 0.010  [4 ] Vm F2 Vm Rk
    0002              122       765       761             0.01        1.005 +- 0.036        0.995 +- 0.036  [3 ] Vm Vm Rk
    0003            12332       618       614             0.01        1.007 +- 0.040        0.994 +- 0.040  [5 ] Vm F2 F2 Vm Rk
    0004           123332        27        33             0.60        0.818 +- 0.157        1.222 +- 0.213  [6 ] Vm F2 F2 F2 Vm Rk
    0005               22         6        11             0.00        0.545 +- 0.223        1.833 +- 0.553  [2 ] Vm Vm
    0006            12232         6         1             0.00        6.000 +- 2.449        0.167 +- 0.167  [5 ] Vm F2 Vm Vm Rk
    0007          1233332         4         2             0.00        2.000 +- 1.000        0.500 +- 0.354  [7 ] Vm F2 F2 F2 F2 Vm Rk
    0008              332         3         1             0.00        3.000 +- 1.732        0.333 +- 0.333  [3 ] Vm F2 F2
    0009            12322         2         1             0.00        2.000 +- 1.414        0.500 +- 0.500  [5 ] Vm Vm F2 Vm Rk
    0010       3333333332         1         1             0.00        1.000 +- 1.000        1.000 +- 1.000  [10] Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0011           123322         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] Vm Vm F2 F2 Vm Rk
    0012             1222         0         1             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] Vm Vm Vm Rk
    .                             100000    100000         0.62/4 =  0.16  (pval:0.960 prob:0.040)  
                     /tmp/blyth/opticks/evt/tboolean-box/torch/1 eda9ec2c306dc7db41b7b5a77461e617 ca967ba3a6c7edbaa805fb212269bf48  100000    -1.0000 INTEROP_MODE 
    {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=100000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'resolution': u'20', u'emit': -1}
    [2017-11-16 14:50:48,791] p17316 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive




::

    tboolean-;tboolean-box --okg4 --dbgseqmat 0x123   ## dump the "F2 Vm Rk"


Using 3 boxes, vacuum/water/glass Vm/Wt/F2 reveals the matswap issue only occurs for a BR happens on the 1st step::

    container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container="1" )  # no param, container="1" switches on auto-sizing


    wbox = CSG("box3", param=[400,400,300,0], emit=0,  boundary="Vacuum///Water" )
    box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Water///GlassSchottF2" )

    CSG.Serialize([container, wbox, box], args.csgpath )


::

    [2017-11-16 14:06:56,465] p15602 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171116-1406 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy 
    B tboolean-box/torch/ -1 :  20171116-1406 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///Water,Water///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000         3.79/9 =  0.42  (pval:0.924 prob:0.076)  
    0000               8d     88711     88704             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] TO SA
    0001           8ccccd      5801      5783             0.03        1.003 +- 0.013        0.997 +- 0.013  [6 ] TO BT BT BT BT SA
    0002             8ccd      4609      4590             0.04        1.004 +- 0.015        0.996 +- 0.015  [4 ] TO BT BT SA

    0003              8bd       258       264             0.07        0.977 +- 0.061        1.023 +- 0.063  [3 ] TO BR SA
        
    0004            8cbcd       160       176             0.76        0.909 +- 0.072        1.100 +- 0.083  [5 ] TO BT BR BT SA
    0005        8cccbcccd       129       120             0.33        1.075 +- 0.095        0.930 +- 0.085  [9 ] TO BT BT BT BR BT BT BT SA
    0006              4cd       110       123             0.73        0.894 +- 0.085        1.118 +- 0.101  [3 ] TO BT AB
    0007              86d       111       110             0.00        1.009 +- 0.096        0.991 +- 0.094  [3 ] TO SC SA
    0008          8ccbccd        49        61             1.31        0.803 +- 0.115        1.245 +- 0.159  [7 ] TO BT BT BR BT BT SA
    0009            4cccd        21        26             0.53        0.808 +- 0.176        1.238 +- 0.243  [5 ] TO BT BT BT AB
    .                             100000    100000         3.79/9 =  0.42  (pval:0.924 prob:0.076)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             100000    100000       407.43/9 = 45.27  (pval:0.000 prob:1.000)  
    0000               23     88711     88704             0.00        1.000 +- 0.003        1.000 +- 0.003  [2 ] Vm Rk
    0001           231413      5801      5783             0.03        1.003 +- 0.013        0.997 +- 0.013  [6 ] Vm Wt F2 Wt Vm Rk
    0002             2313      4609      4590             0.04        1.004 +- 0.015        0.996 +- 0.015  [4 ] Vm Wt Vm Rk

    0003              233       369       110           140.04        3.355 +- 0.175        0.298 +- 0.028  [3 ] Vm Vm Rk
    0004              231         0       264           264.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] Wt Vm Rk

    0005            23113       161       176             0.67        0.915 +- 0.072        1.093 +- 0.082  [5 ] Vm Wt Wt Vm Rk
    0006        231411413       129       120             0.33        1.075 +- 0.095        0.930 +- 0.085  [9 ] Vm Wt F2 Wt Wt F2 Wt Vm Rk
    0007              113       110       123             0.73        0.894 +- 0.085        1.118 +- 0.101  [3 ] Vm Wt Wt
    0008          2314413        51        62             1.07        0.823 +- 0.115        1.216 +- 0.154  [7 ] Vm Wt F2 F2 Wt Vm Rk
    0009            11413        21        26             0.53        0.808 +- 0.176        1.238 +- 0.243  [5 ] Vm Wt F2 Wt Wt
    .                             100000    100000       407.43/9 = 45.27  (pval:0.000 prob:1.000)  

::

    tboolean-;tboolean-box --okg4 --dbgseqmat 0x231   ## dump the 0x231  "Wt Vm Rk"

    // set nowrap

    2017-11-16 14:26:37.292 INFO  [5246439] [CRec::dump@68] CDebug::dump record_id 90234  origin[ -192.665171.235-599.900]  nstp 3  Ori[ -192.665171.235-599.900] 
    ( 0)  TO/BR     FrR              PRE_SAVE POST_SAVE MAT_SWAP STEP_START 
    [   0](Stp ;opticalphoton stepNum    3(tk ;opticalphoton tid 235 pid 0 nm    380 mm  ori[ -192.665 171.235-599.900]  pos[    0.000   0.000  -0.100]  )
      pre                  box_pv0_          Vacuum          noProc           Undefined pos[      0.000     0.000     0.000]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.200 nm 380.000 mm/ns 299.792
     post                 box3_pv1_           Water  Transportation        GeomBoundary pos[      0.000     0.000   449.900]  dir[    0.000   0.000  -1.000]  pol[   -0.000   1.000  -0.000]  ns  1.701 nm 380.000 mm/ns 299.792
     )
    ( 1)  BR/NA     STS                                           POST_SKIP 
    [   1](Stp ;opticalphoton stepNum    3(tk ;opticalphoton tid 235 pid 0 nm    380 mm  ori[ -192.665 171.235-599.900]  pos[    0.000   0.000  -0.100]  )
      pre                 box3_pv1_           Water  Transportation        GeomBoundary pos[      0.000     0.000   449.900]  dir[    0.000   0.000  -1.000]  pol[   -0.000   1.000  -0.000]  ns  1.701 nm 380.000 mm/ns 299.792
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[      0.000     0.000   449.900]  dir[    0.000   0.000  -1.000]  pol[    0.000   1.000   0.000]  ns  1.701 nm 380.000 mm/ns 299.792
     )
    ( 2)  NA/SA     Abs              POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [   2](Stp ;opticalphoton stepNum    3(tk ;opticalphoton tid 235 pid 0 nm    380 mm  ori[ -192.665 171.235-599.900]  pos[    0.000   0.000  -0.100]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[      0.000     0.000   449.900]  dir[    0.000   0.000  -1.000]  pol[    0.000   1.000   0.000]  ns  1.701 nm 380.000 mm/ns 299.792
     post               UNIVERSE_PV            Rock  Transportation        GeomBoundary pos[      0.000     0.000    -0.100]  dir[    0.000   0.000  -1.000]  pol[    0.000   1.000   0.000]  ns  3.202 nm 380.000 mm/ns 299.792
     )




ISSUE material bookeeping difference ? for "TO BR SA" 
--------------------------------------------------------

Observations

* starting with F2 is impossible 


::

    simon:opticks blyth$ tboolean-;tboolean-box-p


    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000      7437.66/7 = 1062.52  (pval:0.000 prob:1.000)  
    0000     532969    532969             0.00  Vm Rk
    0001      58492     58490             0.00  Vm F2 Vm Rk

    0002       4589       483          3323.98  Vm Vm Rk
    0003          0      4113          4113.00  F2 Vm Rk
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ looks like an impossible material history from G4

    0004       3606      3593             0.02  Vm F2 F2 Vm Rk
    0005        213       224             0.28  Vm F2 F2 F2 Vm Rk 

    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.74/7 =  0.11  (pval:0.998 prob:0.002)  
    0000     532969    532969             0.00  TO SA
    0001      58492     58490             0.00  TO BT BT SA
    0002       4107      4113             0.00  TO BR SA
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 



Apply seqhis selection, and look at the seqmat::

    simon:opticks blyth$ tboolean-;tboolean-box-ip

    In [2]: ab.sel = "TO BR SA"

    In [3]: ab
    Out[3]: 
    AB(1,torch,tboolean-box)  TO BR SA 0 
    A tboolean-box/torch/  1 :  20171113-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy 
    B tboolean-box/torch/ -1 :  20171113-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--

    In [4]: ab.mat
    Out[4]: 
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                               4107      4113      8220.00/1 = 8220.00  (pval:0.000 prob:1.000)  
    0000          0      4113          4113.00  F2 Vm Rk
    0001       4107         0          4107.00  Vm Vm Rk
    .                               4107      4113      8220.00/1 = 8220.00  (pval:0.000 prob:1.000)  



::

    In [10]: ab.selmat = "F2 Vm Rk"
    [2017-11-13 17:08:27,699] p54890 {/Users/blyth/opticks/ana/evt.py:742} WARNING - _init_selection EMPTY nsel 0 len(psel) 600000 

    In [11]: ab.his
    Out[11]: 
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                                  0      4113      4113.00/0 = 4113.00  (pval:nan prob:nan)  
    0000          0      4113          4113.00  TO BR SA
    .                                  0      4113      4113.00/0 = 4113.00  (pval:nan prob:nan)  



::

    simon:issues blyth$ tboolean-;tboolean-box-a 
    2017-11-13 17:12:13.441 INFO  [4647617] [Opticks::dumpArgs@816] Opticks::configure argc 10
      0 : OpticksEventCompareTest
      1 : --torch
      2 : --tag
      3 : 1
      4 : --cat
      5 : tboolean-box
      6 : --dbgnode
      7 : 0
      8 : --dbgseqhis
      9 : 0x8bd


    # G4 events (CRecorder?) yielding 321 when 221 expected 


    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventCompare::dumpMatchedSeqHis@67] OpticksEventCompare::dumpMatchedSeqHis A 1
    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventDump::dump@79]  tagdir /tmp/blyth/opticks/evt/tboolean-box/torch/1 photon_id 97
    (      -74.23      98.16    -449.90         0.20)       (    0.00  -1.00   0.00   378.90)               2       3     254      13    TORCH          ?         ?
    (      -74.23      98.16     -99.99         1.37)       (    0.00   1.00   0.00   378.90)               2       1       1      11BOUNDARY_REFLECT          ?         ?
    (      -74.23      98.16    -450.00         2.53)       (    0.00   1.00   0.00   378.90)               1       1       1       8SURFACE_ABSORB          ?         ?
     ph       97   ux 3264509732   fxyzw    -74.233     98.166   -450.000      2.535 
    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventCompare::dumpMatchedSeqHis@76] OpticksEventCompare::dumpMatchedSeqHis B 1
    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventDump::dump@79]  tagdir /tmp/blyth/opticks/evt/tboolean-box/torch/-1 photon_id 162
    (     -133.76     -72.37    -449.90         0.20)       (    0.00  -1.00   0.00   378.90)               3       0       0      13    TORCH          ?         ?
    (     -133.76     -72.37     -99.99         1.37)       (    0.00   1.00   0.00   378.90)               2       0       0      11BOUNDARY_REFLECT          ?         ?
    (     -133.76     -72.37    -450.00         2.53)       (    0.00   1.00   0.00   378.90)               1       0       0       8SURFACE_ABSORB          ?         ?
     ph      162   ux 3271934652   fxyzw   -133.761    -72.366   -450.000      2.535 
    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventCompare::dumpMatchedSeqHis@67] OpticksEventCompare::dumpMatchedSeqHis A 2
    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventDump::dump@79]  tagdir /tmp/blyth/opticks/evt/tboolean-box/torch/1 photon_id 217
    (      135.97    -143.89    -449.90         0.20)       (    0.00  -1.00   0.00   378.90)               2       3     254      13    TORCH          ?         ?
    (      135.97    -143.89     -99.99         1.37)       (    0.00   1.00   0.00   378.90)               2       1       1      11BOUNDARY_REFLECT          ?         ?
    (      135.97    -143.89    -450.00         2.53)       (    0.00   1.00   0.00   378.90)               1       1       1       8SURFACE_ABSORB          ?         ?
     ph      217   ux 1124596356   fxyzw    135.979   -143.885   -450.000      2.535 
    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventCompare::dumpMatchedSeqHis@76] OpticksEventCompare::dumpMatchedSeqHis B 2
    2017-11-13 17:12:17.191 INFO  [4647617] [OpticksEventDump::dump@79]  tagdir /tmp/blyth/opticks/evt/tboolean-box/torch/-1 photon_id 358
    (      -19.99    -135.51    -449.90         0.20)       (    0.00  -1.00   0.00   378.90)               3       0       0      13    TORCH          ?         ?
    (      -19.99    -135.51     -99.99         1.37)       (    0.00   1.00   0.00   378.90)               2       0       0      11BOUNDARY_REFLECT          ?         ?
    (      -19.99    -135.51    -450.00         2.53)       (    0.00   1.00   0.00   378.90)               1       0       0       8SURFACE_ABSORB          ?         ?
     ph      358   ux 3248482992   fxyzw    -19.990   -135.507   -450.000      2.535 
    2017-11-13 17:12:17.192 INFO  [4647617] [OpticksEventCompare::dumpMatchedSeqHis@76] OpticksEventCompare::dumpMatchedSeqHis B 3
    2017-11-13 17:12:17.192 INFO  [4647617] [OpticksEventDump::dump@79]  tagdir /tmp/blyth/opticks/evt/tboolean-box/torch/-1 photon_id 590
    (      -93.32     -34.48    -449.90         0.20)       (    0.00  -1.00   0.00   378.90)               3       0       0      13    TORCH          ?         ?
    (      -93.32     -34.48     -99.99         1.37)       (    0.00   1.00   0.00   378.90)               2       0       0      11BOUNDARY_REFLECT          ?         ?
    (      -93.32     -34.48    -450.00         2.53)       (    0.00   1.00   0.00   378.90)               1       0       0       8SURFACE_ABSORB          ?         ?
     ph      590   ux 3267010736   fxyzw    -93.314    -34.481   -450.000      2.535 



::

     499 #ifdef USE_CUSTOM_BOUNDARY
     500 bool CRecorder::Record(const G4Step* step, int step_id, int record_id, bool dbg, bool other, DsG4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage)
     501 #else
     502 bool CRecorder::Record(const G4Step* step, int step_id, int record_id, bool dbg, bool other, G4OpBoundaryProcessStatus boundary_status, CStage::CStage_t stage)
     503 #endif
     504 {
     505     setStep(step, step_id);
     506     setRecordId(record_id, dbg, other );
     507     setStage(stage);
     508 
     509     LOG(trace) << "CRecorder::Record"
     510               << " step_id " << step_id
     511               << " record_id " << record_id
     512               << " stage " << CStage::Label(stage)
     513               ;
     514 
     515     if(stage == CStage::START)
     516     {
     517         startPhoton();       // MUST be invoked prior to setBoundaryStatus
     518         RecordQuadrant();
     519     }
     520     else if(stage == CStage::REJOIN )
     521     {
     522         if(m_live)
     523         {
     524             decrementSlot();    // this allows REJOIN changing of a slot flag from BULK_ABSORB to BULK_REEMIT 
     525         }
     526         else
     527         {
     528             m_crec->clearStp(); // rejoin happens on output side, not in the crec CStp list
     529         }
     530     }
     531     else if(stage == CStage::RECOLL )
     532     {
     533         m_decrement_request = 0 ;
     534     }
     535 
     536     const G4StepPoint* pre  = m_step->GetPreStepPoint() ;
     537     const G4StepPoint* post = m_step->GetPostStepPoint() ;
     538 
     539     const G4Material* preMat  = pre->GetMaterial() ;
     540     const G4Material* postMat = post->GetMaterial() ;
     541 
     542     unsigned preMaterial = preMat ? m_material_bridge->getMaterialIndex(preMat) + 1 : 0 ;
     543     unsigned postMaterial = postMat ? m_material_bridge->getMaterialIndex(postMat) + 1 : 0 ;
     544 
     545     setBoundaryStatus( boundary_status, preMaterial, postMaterial);
     546 



tboolean-box-p
----------------

::

    simon:opticks blyth$ tboolean-;tboolean-box--
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.polyconfig import PolyConfig
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="/tmp/blyth/opticks/tboolean-box--")

    CSG.kwa = dict(poly="IM",resolution="20", verbosity="0",ctrl="0", containerscale="3", emitconfig="photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1"  )

    container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container="1" )  # no param, container="1" switches on auto-sizing

    box = CSG("box3", param=[300,300,200,0], boundary="Vacuum///GlassSchottF2" )

    CSG.Serialize([container, box], args.csgpath )


    simon:opticks blyth$ tboolean-;tboolean-box-p
    args: /Users/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    ok.smry 1 
    [2017-11-13 16:42:30,204] p54515 {/Users/blyth/opticks/ana/tboolean.py:17} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    [2017-11-13 16:42:30,204] p54515 {/Users/blyth/opticks/ana/ab.py:80} INFO - AB.load START smry 1 
    [2017-11-13 16:42:30,922] p54515 {/Users/blyth/opticks/ana/ab.py:96} INFO - AB.load DONE 
    [2017-11-13 16:42:30,926] p54515 {/Users/blyth/opticks/ana/ab.py:135} INFO - AB.init_point START
    [2017-11-13 16:42:30,929] p54515 {/Users/blyth/opticks/ana/ab.py:137} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171113-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy 
    B tboolean-box/torch/ -1 :  20171113-1638 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.74/7 =  0.11  (pval:0.998 prob:0.002)  
    0000     532969    532969             0.00  TO SA
    0001      58492     58490             0.00  TO BT BT SA
    0002       4107      4113             0.00  TO BR SA
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0003       3602      3590             0.02  TO BT BR BT SA
    0004        482       483             0.00  TO SC SA
    0005        210       222             0.33  TO BT BR BR BT SA
    0006         42        42             0.00  TO AB
    0007         19        23             0.38  TO BT BT SC SA
    0008         16        10             0.00  TO BT BR BR BR BT SA
    0009         12        14             0.00  TO SC BT BR BT SA
    0010         12         6             0.00  TO BT AB
    0011          6        11             0.00  TO SC BT BT SA
    0012          5         6             0.00  TO BT SC BR BR BR BR BR BR BR
    0013          6         4             0.00  TO SC BR SA
    0014          4         3             0.00  TO BT SC BT SA
    0015          3         1             0.00  TO BT SC BR BT SA
    0016          3         2             0.00  TO SC BT BR BR BT SA
    0017          0         3             0.00  TO BT BR AB
    0018          2         0             0.00  TO BT BR BT SC SA
    0019          2         1             0.00  TO BT BT AB
    .                             600000    600000         0.74/7 =  0.11  (pval:0.998 prob:0.002)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         1.57/7 =  0.22  (pval:0.980 prob:0.020)  
    0000     532969    532969             0.00  TO|SA
    0001      58492     58490             0.00  TO|BT|SA
    0002       4107      4113             0.00  TO|BR|SA
    0003       3828      3822             0.00  TO|BT|BR|SA
    0004        482       483             0.00  TO|SA|SC
    0005         42        42             0.00  TO|AB
    0006         29        38             1.21  TO|BT|SA|SC
    0007         25        21             0.35  TO|BT|BR|SA|SC
    0008         14         7             0.00  TO|BT|AB
    0009          5         7             0.00  TO|BT|BR|SC
    0010          7         5             0.00  TO|BR|SA|SC
    0011          0         3             0.00  TO|BT|BR|AB
    .                             600000    600000         1.57/7 =  0.22  (pval:0.980 prob:0.020)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000      7437.66/7 = 1062.52  (pval:0.000 prob:1.000)  
    0000     532969    532969             0.00  Vm Rk
    0001      58492     58490             0.00  Vm F2 Vm Rk
    0002       4589       483          3323.98  Vm Vm Rk
    0003          0      4113          4113.00  F2 Vm Rk
    0004       3606      3593             0.02  Vm F2 F2 Vm Rk
    0005        213       224             0.28  Vm F2 F2 F2 Vm Rk
    0006         42        42             0.00  Vm Vm
    0007         19        23             0.38  Vm F2 Vm Vm Rk
    0008         17        10             0.00  Vm F2 F2 F2 F2 Vm Rk
    0009         12        14             0.00  Vm Vm F2 F2 Vm Rk
    0010         12         6             0.00  Vm F2 F2
    0011          6        11             0.00  Vm Vm F2 Vm Rk
    0012          7         4             0.00  Vm Vm Vm Rk
    0013          5         6             0.00  Vm F2 F2 F2 F2 F2 F2 F2 F2 F2
    0014          3         2             0.00  Vm Vm F2 F2 F2 Vm Rk
    0015          0         3             0.00  Vm F2 F2 F2
    0016          2         1             0.00  Vm F2 Vm Vm
    0017          2         0             0.00  Vm F2 F2 Vm Vm Rk
    0018          1         0             0.00  Vm F2 F2 F2 F2 F2 F2 Vm Rk
    0019          0         1             0.00  Vm Vm F2 F2 F2 F2 F2 F2 F2 Vm
    .                             600000    600000      7437.66/7 = 1062.52  (pval:0.000 prob:1.000)  
                     /tmp/blyth/opticks/evt/tboolean-box/torch/1 2d8f77a2b4cae1dab70b144a03217240 a7ecd069d76241894675465c294c7f30  600000    -1.0000 INTEROP_MODE 
    {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1', u'resolution': u'20', u'emit': -1}
    [2017-11-13 16:42:30,935] p54515 {/Users/blyth/opticks/ana/tboolean.py:25} INFO - early exit as non-interactive
    simon:opticks blyth$ 



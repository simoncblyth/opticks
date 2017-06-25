Analytic Geometry Shakedown
===============================

Issue : flickery ray trace in rib shape
-------------------------------------------

* ~/opticks_refs/tgltf_looking_up_poke_thru_ribs.png


TODO: auto-detect intended unions with coincident faces, and auto-uncoincide them
-----------------------------------------------------------------------------------

* need to implement more parametric ? but thats a lot of work, and
  only very effective at bileaf level when combining primitives, but
  combining composites is very common  

* how about cross bbox scans like NCSGScanTest
  that counts zero crossings : straightforward and works at all
  levels of the CSG tree... for convex solids (most are)
  should be 2 crossings


DONE : Generalized intersect test 
-----------------------------------

* need to compare positions against the implicit func, 
  so need the geometry and event info together

* did this but find the SDF is afflicted by the same issue, iso-zeros 
  at the union join, so contrasting ray trace intersect positiosn against SDF 
  values doesnt help much : actually is does help, it means can test the SDF 
  with scans to find the issues 


Added *--dbgnode IDX* option to pass the OpticksEvent down from okg/OpticksHub into ggeo/GScene 
where apply the inverse global transform to see the local frame positions of the intersects.

::

    2017-06-24 16:51:55.358 INFO  [768643] [GScene::debugNodeIntersects@854]  pho 100000,4,4
    2017-06-24 16:51:55.358 INFO  [768643] [GScene::debugNodeIntersects@855]  seq 100000,1,2
     i   1148 c   1000 seqhis               8d seqmat               31 post vec4(-17525.298828, -801036.750000, -5561.204102, 9.274718) lpos vec4(1423.812500, -261.093750, 1538.795898, 1.000000)
     i   2283 c   2000 seqhis               8d seqmat               31 post vec4(-18471.365234, -800038.937500, -5565.000000, 6.074214) lpos vec4(72.125000, -513.468750, 1535.000000, 1.000000)
     i   3410 c   3000 seqhis               8d seqmat               31 post vec4(-16557.388672, -799992.250000, -7261.199707, 8.752926) lpos vec4(1072.562500, 1118.937500, -161.199707, 1.000000)
     i   4559 c   4000 seqhis               8d seqmat               31 post vec4(-18915.132812, -799739.000000, -5529.183594, 7.106152) lpos vec4(-420.687500, -723.125000, 1570.816406, 1.000000)
     i   5677 c   5000 seqhis               8d seqmat               31 post vec4(-18866.658203, -801034.687500, -5690.345703, 9.352414) lpos vec4(693.500000, -1386.187500, 1409.654297, 1.000000)
     i   6804 c   6000 seqhis               8d seqmat               31 post vec4(-18144.814453, -799436.625000, -5565.000000, 5.624814) lpos vec4(-256.187500, 87.906250, 1535.000000, 1.000000)
     i   7928 c   7000 seqhis               8d seqmat               31 post vec4(-19486.349609, -799048.875000, -6014.063477, 8.627912) lpos vec4(-1310.437500, -827.875000, 1085.936523, 1.000000)

tgltf.py 
----------

Huh, top of cyl-z should not be there::

    In [8]: lpos[lpos[:,2] > 1500 ][:100]
    Out[8]: 
    A()sliced
    A([[ -367.125 ,   236.7812,  1535.    ,     1.    ],
           [  337.    , -1032.    ,  1535.    ,     1.    ],
           [  568.8125, -1328.9688,  1535.    ,     1.    ],
           [ 1212.875 ,  -858.375 ,  1535.    ,     1.    ],
           [  137.0625,  -371.6875,  1535.    ,     1.    ],
           [  849.6875,   997.6562,  1545.9814,     1.    ],
           [ -936.5625,   868.7812,  1547.71  ,     1.    ],
           [  196.3125,   411.9688,  1535.    ,     1.    ],
           [  -55.625 ,  -304.75  ,  1535.    ,     1.    ],
           [ -144.5   ,  -538.3125,  1535.    ,     1.    ],
           [ 1299.0625,  -612.9375,  1535.    ,     1.    ],
           [ -407.5   ,    13.3438,  1535.    ,     1.    ],
           [  865.375 ,   370.4062,  1535.    ,     1.    ],
           [  416.75  ,   478.5938,  1535.    ,     1.    ],
           [  431.75  ,   800.6875,  1535.    ,     1.    ],
           [   -8.5625,  1549.9375,  1526.9644,     1.    ],
           [  948.25  ,  -512.3438,  1535.    ,     1.    ],
           [  229.    ,   -32.5625,  1535.    ,     1.    ],
           [-1007.125 ,  -461.25  ,  1535.    ,     1.    ],
           [  -74.6875,  -607.125 ,  1535.    ,     1.    ],
           [  503.625 ,  -807.9062,  1535.    ,     1.    ],
           [  160.125 , -1057.0625,  1535.    ,     1.    ],
           [ -798.3125,    67.3125,  1535.    ,     1.    ],
           [-1278.25  ,   865.4062,  1535.    ,     1.    ],
           [ -509.625 ,   477.1562,  1535.    ,     1.    ],
           [ -141.875 ,  1289.5   ,  1535.    ,     1.    ],



Hmm, suspect the SDF is afflicted by the same issue... ie iso zero at union join.  
So comparing against SDF doesnt see the problem. But the visualization does...

::

    2017-06-24 18:35:11.940 INFO  [796463] [GScene::debugNodeIntersects@846] OpticksHub::debugNode 3159 epsilon 0.1
    2017-06-24 18:35:11.940 INFO  [796463] [GScene::debugNodeIntersects@847]  nodelib GNodeLib targetnode 0 reldir analytic/GScene/GNodeLib numPV 12230 numLV 12230 numSolids 12230 PV(0) top LV(0) World0xc15cfc0 ( 0 ) ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 6 ) ( 7 ) ( 8 ) ( 9 )
    2017-06-24 18:35:11.940 INFO  [796463] [GScene::debugNodeIntersects@848]  solid 
    2017-06-24 18:35:11.940 INFO  [796463] [GScene::debugNodeIntersects@849]  csg.meta  treeindex -1 depth -1 nchild -1 lvname /dd/Geometry/AD/lvGDS0xbf6cbb8 soname gds0xc28d3f0 isSkip 0 is_uncoincide 1
    2017-06-24 18:35:11.940 INFO  [796463] [GScene::debugNodeIntersects@850]  csg.desc NCSG  index 36 treedir /tmp/blyth/opticks/tgltf/extras/22 node_sh 7,4,4 tran_sh 5,3,4,4 boundary NULL meta NParameters numItems 6         lvname : /dd/Geometry/AD/lvGDS0xbf6cbb8 :          soname :    gds0xc28d3f0 :       verbosity :               0 :      resolution :              20 :            poly :              IM :          height :               2 : 
           gtr  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 

          igtr  0.543   0.840  -0.000  -0.000 
               -0.840   0.543  -0.000   0.000 
                0.000  -0.000   1.000  -0.000 
              -661623.250 449556.188 7100.000   1.000 




    2017-06-24 18:35:11.940 INFO  [796463] [GScene::debugNodeIntersects@866]  pho 100000,4,4
    2017-06-24 18:35:11.940 INFO  [796463] [GScene::debugNodeIntersects@867]  seq 100000,1,2
    2017-06-24 18:35:12.039 INFO  [796463] [GScene::debugNodeIntersects@932] SDF excursions by photon seqhis categories  num_pho 100000 epsilon 0.1
     seqhis               4d tot   5003 exc   5003 exc/tot      1
     seqhis               8d tot  88155 exc      0 exc/tot      0
     seqhis              45d tot    267 exc    267 exc/tot      1
     seqhis              46d tot    187 exc    187 exc/tot      1
     seqhis              85d tot   1743 exc      0 exc/tot      0
     seqhis              86d tot   3583 exc      0 exc/tot      0
     seqhis             455d tot     59 exc     59 exc/tot      1
     seqhis             456d tot      4 exc      4 exc/tot      1
     seqhis             465d tot      4 exc      4 exc/tot      1
     seqhis             466d tot      2 exc      2 exc/tot      1
     seqhis             855d tot    450 exc      0 exc/tot      0
     seqhis             856d tot     73 exc      0 exc/tot      0
     seqhis             865d tot     62 exc      0 exc/tot      0
     seqhis             866d tot    141 exc      0 exc/tot      0
     seqhis            4555d tot     21 exc     21 exc/tot      1
     seqhis            4556d tot      3 exc      3 exc/tot      1
     seqhis            4655d tot      3 exc      3 exc/tot      1
     seqhis            8555d tot    115 exc      0 exc/tot      0
     seqhis            8556d tot     16 exc      0 exc/tot      0
     seqhis            8565d tot      2 exc      0 exc/tot      0
     seqhis            8566d tot      2 exc      0 exc/tot      0
     seqhis            8655d tot     15 exc      0 exc/tot      0
     seqhis            8656d tot      6 exc      0 exc/tot      0
     seqhis            8665d tot      2 exc      0 exc/tot      0
     seqhis            8666d tot      6 exc      0 exc/tot      0
     seqhis           45555d tot      7 exc      7 exc/tot      1
     seqhis           45556d tot      1 exc      1 exc/tot      1
     seqhis           46556d tot      2 exc      2 exc/tot      1
     seqhis           85555d tot     39 exc      0 exc/tot      0
     seqhis           85556d tot      3 exc      0 exc/tot      0
     seqhis           85565d tot      1 exc      0 exc/tot      0
     seqhis           85566d tot      1 exc      0 exc/tot      0
     seqhis           85655d tot      1 exc      0 exc/tot      0
     seqhis           85656d tot      1 exc      0 exc/tot      0
     seqhis           86555d tot      4 exc      0 exc/tot      0
     seqhis           86565d tot      1 exc      0 exc/tot      0
     seqhis          455555d tot      1 exc      1 exc/tot      1
     seqhis          855555d tot      7 exc      0 exc/tot      0




NCSGScanTest
---------------

::

    delta:ggeo blyth$ grep gds /tmp/blyth/opticks/tgltf/extras/*/meta.json 
    /tmp/blyth/opticks/tgltf/extras/22/meta.json:{"lvname": "/dd/Geometry/AD/lvGDS0xbf6cbb8", "soname": "gds0xc28d3f0", "verbosity": "0", "resolution": "20", "poly": "IM", "height": 2}
    delta:ggeo blyth$ 


Confirmed iso-zero at union join::

    elta:opticks blyth$ SCAN=0,0,1500,0,0,1,0,200,5 NCSGScanTest /tmp/blyth/opticks/tgltf/extras/22
    2017-06-24 20:27:06.333 INFO  [815415] [main@30]  argc 2 argv[0] NCSGScanTest
    BStr::fsplitEnv envvar SCAN line 0,0,1500,0,0,1,0,200,5 fallback 0,0,128,0,0,1,-1,1,0.001 elem.size 9
    2017-06-24 20:27:06.335 INFO  [815415] [nnode::Scan@364] nnode::Scan
     origin {    0.0000    0.0000 1500.0000} direction {    0.0000    0.0000    1.0000} range {    0.0000  200.0000    5.0000}
     t     0.0000 x     0.0000 y     0.0000 z  1500.0000 :   -35.0000
     t     5.0000 x     0.0000 y     0.0000 z  1505.0000 :   -30.0000
     t    10.0000 x     0.0000 y     0.0000 z  1510.0000 :   -25.0000
     t    15.0000 x     0.0000 y     0.0000 z  1515.0000 :   -20.0000
     t    20.0000 x     0.0000 y     0.0000 z  1520.0000 :   -15.0000
     t    25.0000 x     0.0000 y     0.0000 z  1525.0000 :   -10.0000
     t    30.0000 x     0.0000 y     0.0000 z  1530.0000 :    -5.0000
     t    35.0000 x     0.0000 y     0.0000 z  1535.0000 :     0.0000
     t    40.0000 x     0.0000 y     0.0000 z  1540.0000 :    -5.0000
     t    45.0000 x     0.0000 y     0.0000 z  1545.0000 :   -10.0000
     t    50.0000 x     0.0000 y     0.0000 z  1550.0000 :   -15.0000
     t    55.0000 x     0.0000 y     0.0000 z  1555.0000 :   -20.0000
     t    60.0000 x     0.0000 y     0.0000 z  1560.0000 :   -25.0000
     t    65.0000 x     0.0000 y     0.0000 z  1565.0000 :   -30.0000
     t    70.0000 x     0.0000 y     0.0000 z  1570.0000 :   -35.0000
     t    75.0000 x     0.0000 y     0.0000 z  1575.0000 :   -35.7292
     t    80.0000 x     0.0000 y     0.0000 z  1580.0000 :   -30.7292
     t    85.0000 x     0.0000 y     0.0000 z  1585.0000 :   -25.7292
     t    90.0000 x     0.0000 y     0.0000 z  1590.0000 :   -20.7292
     t    95.0000 x     0.0000 y     0.0000 z  1595.0000 :   -15.7292
     t   100.0000 x     0.0000 y     0.0000 z  1600.0000 :   -10.7292
     t   105.0000 x     0.0000 y     0.0000 z  1605.0000 :    -5.7292
     t   110.0000 x     0.0000 y     0.0000 z  1610.0000 :    -0.7292
     t   115.0000 x     0.0000 y     0.0000 z  1615.0000 :    -4.2708
     t   120.0000 x     0.0000 y     0.0000 z  1620.0000 :    -4.4397
     t   125.0000 x     0.0000 y     0.0000 z  1625.0000 :     0.5603
     t   130.0000 x     0.0000 y     0.0000 z  1630.0000 :     5.5603
     t   135.0000 x     0.0000 y     0.0000 z  1635.0000 :    10.5603
     t   140.0000 x     0.0000 y     0.0000 z  1640.0000 :    15.5603
     t   145.0000 x     0.0000 y     0.0000 z  1645.0000 :    20.5603
     t   150.0000 x     0.0000 y     0.0000 z  1650.0000 :    25.5603
     t   155.0000 x     0.0000 y     0.0000 z  1655.0000 :    30.5603
     t   160.0000 x     0.0000 y     0.0000 z  1660.0000 :    35.5603
     t   165.0000 x     0.0000 y     0.0000 z  1665.0000 :    40.5603
     t   170.0000 x     0.0000 y     0.0000 z  1670.0000 :    45.5603
     t   175.0000 x     0.0000 y     0.0000 z  1675.0000 :    50.5603
     t   180.0000 x     0.0000 y     0.0000 z  1680.0000 :    55.5603
     t   185.0000 x     0.0000 y     0.0000 z  1685.0000 :    60.5603
     t   190.0000 x     0.0000 y     0.0000 z  1690.0000 :    65.5603
     t   195.0000 x     0.0000 y     0.0000 z  1695.0000 :    70.5603
     t   200.0000 x     0.0000 y     0.0000 z  1700.0000 :    75.5603
    delta:opticks blyth$ 




Issue : upward yellow cone lid photons think start in Ac
----------------------------------------------------------

* Many ~5% photons (all upward cone) think they are in acrylic.
* BUT, the yellow photons are spread around, not all pointing at poke thru ribs

::

    In [4]: print a.mat[:10]
    .                             1:gltf 
    .                             100000         1.00 
    0000           343231        0.460       45953      [6 ] Gd Ac LS Ac MO Ac
    0001          aa33231        0.047        4728      [7 ] Gd Ac LS Ac Ac ES ES
    0002               11        0.044        4396      [2 ] Gd Gd
    0003          3432311        0.026        2563      [7 ] Gd Gd Ac LS Ac MO Ac
    0004          5d43231        0.024        2406      [7 ] Gd Ac LS Ac MO Vm Bk
    0005               33        0.024        2385      [2 ] Ac Ac
    0006       4323233133        0.023        2345      [10] Ac Ac Gd Ac Ac LS Ac LS Ac MO
    0007          aa34231        0.018        1819      [7 ] Gd Ac LS MO Ac ES ES
    0008       3432323133        0.015        1461      [10] Ac Ac Gd Ac LS Ac LS Ac MO Ac
    0009         5de43231        0.011        1129      [8 ] Gd Ac LS Ac MO Py Vm Bk
    .                             100000         1.00 



* tboolean-gds obtained from CSG code generation, nd.mesh.csg.dump_tboolean("gds")
  does not exhibit the issue... 

  * the only difference is the top level transform (ie dont get issue when near origin) ?? 
    suggests a numerical issue with small nudges ? 


::

    In [8]: nd = sc.get_node(3159)

    In [9]: print nd.mesh.csg.txt
        un abc            
    cy a         un bc    
            co b     cy c

    In [10]: nd.
    nd.brief       nd.depth       nd.find_nodes  nd.matrix      nd.name        nd.parent      nd.soIdx       
    nd.children    nd.extras      nd.gltf        nd.mesh        nd.ndIdx       nd.scene       nd.transform   

    In [10]: nd.gltf
    Out[10]: 
    {'extras': {'boundary': 'Acrylic///GdDopedLS',
      'pvname': '/dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00',
      'selected': 1},
     'matrix': [1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
      7.5,
      1.0],
     'mesh': 36,
     'name': '/dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00'}



GScene::dumpNode
~~~~~~~~~~~~~~~~~~~

::

    2017-06-24 11:18:39.577 INFO  [623565] [GScene::dumpNode@69] GScene::dump_node nidx   3158 FOUND 
    nd idx/repeatIdx/mesh/nch/depth/nprog  [3158:  0: 35:  2:13:   2] bnd:LiquidScintillator///Acrylic   
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   2.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7107.500   1.000 


    2017-06-24 11:18:39.577 INFO  [623565] [GScene::dumpNode@69] GScene::dump_node nidx   3159 FOUND 
    nd idx/repeatIdx/mesh/nch/depth/nprog  [3159:  0: 36:  0:14:   0] bnd:Acrylic///GdDopedLS   
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   7.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 



::

    In [2]: nd.gtr_mdot_r
    Out[2]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [ -18079.4531, -799699.4375,   -7100.    ,       1.    ]], dtype=float32)

    In [3]: nd.gtr_mdotr_r
    Out[3]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [ -18079.4531, -799699.4375,   -7100.    ,       1.    ]], dtype=float32)

    In [4]: 

    In [4]: nd.gtr_mdotr
    Out[4]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [  19391.    ,  802110.    ,   -7100.    ,       1.    ]], dtype=float32)

    In [5]: nd.gtr_mdot
    Out[5]: 
    array([[      0.5432,      -0.8396,       0.    ,       0.    ],
           [      0.8396,       0.5432,       0.    ,       0.    ],
           [      0.    ,       0.    ,       1.    ,       0.    ],
           [  19391.    ,  802110.    ,   -7100.    ,       1.    ]], dtype=float32)






Approach ? Decide to implement recursive geo selection to onion in on the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplify... test with just the GdLS


* succeed to reproduce with, 2 volumes (presumably polycone concidence issue)
  (need an outer)

::

    export OPTICKS_QUERY="range:3158:3160"   # 3158+3159
    #export OPTICKS_QUERY="index:3159,depth:2"


::

    [2017-06-23 19:50:29,366] p36145 {/Users/blyth/opticks/ana/OpticksQuery.py:75} INFO - index found at depth 14 
    [2017-06-23 19:50:29,367] p36145 {/Users/blyth/opticks/analytic/treebase.py:216} INFO - selected index  3159 depth 14 name /dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00 mat GdDopedLS
    [2017-06-23 19:50:29,387] p36145 {/Users/blyth/opticks/analytic/treebase.py:501} INFO - apply_selection OpticksQuery index:3159,depth:2 range [] index 3159 depth 2   Node.selected_count 1 


* ~/opticks_refs/tachyon_reflection_from_top_3159.png 

::

    3157      3156 [ 11:   0/ 520]    3 ( 0)              __dd__Geometry__AD__lvOAV0xbf1c760  oav0xc2ed7c8
    3158      3157 [ 12:   0/   3]   35 ( 0)               __dd__Geometry__AD__lvLSO0xc403e40  lso0xc028a38
    3159      3158 [ 13:   0/  35]    2 ( 0)                __dd__Geometry__AD__lvIAV0xc404ee8  iav0xc346f90
    3160      3159 [ 14:   0/   2]    0 ( 0)                 __dd__Geometry__AD__lvGDS0xbf6cbb8  gds0xc28d3f0
    3161      3160 [ 14:   1/   2]    0 ( 0)                 __dd__Geometry__AdDetails__lvOcrGdsInIav0xbf6dd58  OcrGdsInIav0xc405b10
    3162      3161 [ 13:   1/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvIavTopHub0xc129d88  IavTopHub0xc405968
    3163      3162 [ 13:   2/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0  CtrGdsOflBotClp0xbf5dec0
    3164      3163 [ 13:   3/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflTfbInLso0xbfa0728  CtrGdsOflTfbInLso0xbfa2d30
    3165      3164 [ 13:   4/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflInLso0xc28cc88  CtrGdsOflInLso0xbfa1178
    3166      3165 [ 13:   5/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsPrt0xc352630  OcrGdsPrt0xc352518
    3167      3166 [ 13:   6/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0  CtrGdsOflBotClp0xbf5dec0
    3168      3167 [ 13:   7/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsTfbInLso0xc3529c0  OcrGdsTfbInLso0xbfa2370
    3169      3168 [ 13:   8/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsInLso0xc353990  OcrGdsInLso0xbfa2190
    3170      3169 [ 13:   9/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOavBotRib0xc353d30  OavBotRib0xbfaafe0
    3171      3170 [ 13:  10/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOavBotRib0xc353d30  OavBotRib0xbfaafe0



::

    In [12]: c.mesh.csg.dump(detailed=True)
    [2017-06-23 20:29:35,658] p37472 {/Users/blyth/opticks/analytic/csg.py:783} INFO - CSG.dump name:gds0xc28d3f0
    un(cy,un(co,cy) height:1 totnodes:3 ) height:2 totnodes:7 
     union;gds0xc28d3f0                                : abc = CSG("union", left=a, right=bc) 
        cylinder;gds_cyl0xc570d78_outer                : a = CSG("cylinder", param = [0.000,0.000,0.000,1550.000],param1 = [-1535.000,1535.000,0.000,0.000]) 
        union;gds_polycone0xc404f40_uniontree          : bc = CSG("union", left=b, right=c) 
           cone;gds_polycone0xc404f40_zp_1             : b = CSG("cone", param = [1520.000,3070.000,75.000,3145.729],param1 = [0.000,0.000,0.000,0.000]) 
           cylinder;gds_polycone0xc404f40_zp_2         : c = CSG("cylinder", param = [0.000,0.000,0.000,75.000],param1 = [3145.729,3159.440,0.000,0.000]) 

        un abc            
    cy a         un bc    
            co b     cy c



::

    In [15]: c.mesh.csg.dump_tboolean("gds")


    tboolean-gds(){ TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
    tboolean-gds-(){ $FUNCNAME- | python $* ; } 
    tboolean-gds--(){ cat << EOP

    outdir = "$TMP/$FUNCNAME"
    obj_ = "$(tboolean-testobject)"
    con_ = "$(tboolean-container)"

    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main()

    CSG.boundary = obj_
    CSG.kwa = dict(verbosity="1")

    a = CSG("cylinder", param = [0.000,0.000,0.000,1550.000],param1 = [-1535.000,1535.000,0.000,0.000])

    b = CSG("cone", param = [1520.000,3070.000,75.000,3145.729],param1 = [0.000,0.000,0.000,0.000])  # r1,z1,r2,z2  
    c = CSG("cylinder", param = [0.000,0.000,0.000,75.000],param1 = [3145.729,3159.440,0.000,0.000])
    bc = CSG("union", left=b, right=c)
    bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1535.000,1.000]]

    abc = CSG("union", left=a, right=bc)

    obj = abc

    con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=con_ , poly="HY", level="5" )
    CSG.Serialize([con, obj], outdir )


    EOP
    }




::

      519     <tube aunit="deg" deltaphi="360" lunit="mm" name="gds_cyl0xc570d78" rmax="1550" rmin="0" startphi="0" z="3070"/>
      520     <polycone aunit="deg" deltaphi="360" lunit="mm" name="gds_polycone0xc404f40" startphi="0">
      521       <zplane rmax="1520" rmin="0" z="3070"/>
      522       <zplane rmax="75" rmin="0" z="3145.72924106399"/>
      523       <zplane rmax="75" rmin="0" z="3159.43963177189"/>
      524     </polycone>
      525     <union name="gds0xc28d3f0">
      526       <first ref="gds_cyl0xc570d78"/>
      527       <second ref="gds_polycone0xc404f40"/>
      528       <position name="gds0xc28d3f0_pos" unit="mm" x="0" y="0" z="-1535"/>
      529     </union>




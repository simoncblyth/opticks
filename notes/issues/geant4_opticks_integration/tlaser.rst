tlaser
========

Shoot horizontal laser in X direction (vertical geometry too involved)::

     19 tlaser--(){
     21     local msg="=== $FUNCNAME :"
     23     local det=$(tlaser-det)
     24     local tag=$(tlaser-tag)
     25 
     26     local torch_config=(
     27                  type=point
     28                  frame=3153
     29                  source=0,0,0
     30                  target=1,0,0
     31                  photons=10000
     32                  material=GdDopedLS
     33                  wavelength=430
     34                  weight=1.0
     35                  time=0.1
     36                  zenithazimuth=0,1,0,1
     37                  radius=0
     38                )
     40     op.sh  \
     41             $* \
     42             --animtimemax 15 \
     43             --timemax 15 \
     44             --eye 0,1,0 \
     45             --torch --torchconfig "$(join _ ${torch_config[@]})" \
     46             --torchdbg \
     47             --save --tag $tag --cat $det
     51 }

::

    tlaser- ; tlaser-- --okg4 --compute



Prior to fixing aim
----------------------


::
    delta:ana blyth$ tlaser.py  ## apply seqhis selection to pick the most common seqs for A and B

      A:seqhis_ana       noname 
              8ccccd        1.000           7673       [6 ] TO BT BT BT BT SA
                            7673         1.00 
       B:seqhis_ana       noname 
            8c0cc0cd        1.000           7030       [8 ] TO BT ?0? BT BT ?0? BT SA
                            7030         1.00 



Laser aim issue
-------------------

Huh looks like laser going in different directions::

    In [6]: a.rpost_(slice(0,6))     ## heading in some combination of X and Y direction
    Out[6]: 
    A()sliced
    A([[[ -18079.4443, -799699.4149,   -6604.9499,       0.0998],
            [ -17219.8321, -800985.8917,   -6604.9499,       7.8266],
            [ -17214.1845, -800994.1278,   -6604.9499,       7.8765],
            [ -16980.2796, -801344.2792,   -6604.9499,       9.98  ],
            [ -16970.161 , -801359.3395,   -6604.9499,      10.0702],
            [ -16826.3825, -801575.3603,   -6604.9499,      11.3474]],

       In [13]: b.rpost_(slice(0,6))   ## huh heading in -Z direction
    Out[13]: 
    A()sliced
    A([[[ -18079.4443, -799699.4149,   -6604.9499,       0.0998],
            [ -18079.4443, -799699.4149,   -8635.0278,      10.5229],
            [ -18079.4443, -799699.4149,   -8650.0881,      10.6008],
            [ -18079.4443, -799699.4149,   -8850.1073,      11.639 ],
            [ -18079.4443, -799699.4149,   -8895.0528,      11.8702],
            [ -18079.4443, -799699.4149,   -9092.013 ,      12.8928]],

::

    OKTest --load --vizg4 --cat laser
    OKG4Test --load --vizg4 --cat laser
    

Gensteps are same by construction, suspect CTorchSource not reading it::

    In [3]: a.gs
    Out[3]: 
    A(torch,1,laser)-
    A([[[      0.    ,       0.    ,       0.    ,       0.    ],
            [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.5556,      -0.8314,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,     430.    ],
            [      0.    ,       1.    ,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]]], dtype=float32)

    In [4]: b.gs
    Out[4]: 
    A(torch,-1,laser)-
    A([[[      0.    ,       0.    ,       0.    ,       0.    ],
            [ -18079.4531, -799699.4375,   -6605.    ,       0.1   ],
            [      0.5556,      -0.8314,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,     430.    ],
            [      0.    ,       1.    ,       0.    ,       1.    ],
            [      0.    ,       0.    ,       0.    ,       0.    ]]], dtype=float32)



after fix aiming, restricted to top seq
--------------------------------------------

Restricting to top seq::

      A:seqhis_ana       noname 
              8ccccd        1.000           7673       [6 ] TO BT BT BT BT SA
                            7673         1.00 
       B:seqhis_ana       noname 
            8ccccccd        1.000           7500       [8 ] TO BT BT BT BT BT BT SA
                            7500         1.00 


       tlaser- ; tlaser-- --okg4 --compute --dbgseqhis 8ccccccd


::

    In [8]: a.rpost_(slice(0,9))[0]
    Out[8]: 
    A()sliced
    A([[     -18079.4443, -799699.4149,   -6604.9499,       0.0998],
           [ -17219.8321, -800985.8917,   -6604.9499,       7.8266],
           [ -17214.1845, -800994.1278,   -6604.9499,       7.8765],
           [ -16980.2796, -801344.2792,   -6604.9499,       9.98  ],
           [ -16970.161 , -801359.3395,   -6604.9499,      10.0702],
           [ -16826.3825, -801575.3603,   -6604.9499,      11.3474],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],   << decompression dummies
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ]])


    In [14]: a.ox[:,0]    # final position photons, no compression
    Out[14]: 
    A()sliced
    A([[ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           ..., 
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472],
           [ -16826.3945, -801575.375 ,   -6605.    ,      11.3472]], dtype=float32)



    In [9]: b.rpost_(slice(0,9))[0]
    Out[9]: 
    A()sliced
    A([[     -18079.4443, -799699.4149,   -6604.9499,       0.0998],
           [ -17218.1849, -800988.2449,   -6604.9499,       8.0587],
           [ -17212.7726, -800996.481 ,   -6604.9499,       8.1104],
           [ -16978.1618, -801347.3383,   -6604.9499,      10.2771],
           [ -16968.2785, -801362.3986,   -6604.9499,      10.3705],
           [ -16824.2646, -801577.7134,   -6604.9499,      11.6829],
           [ -16822.6174, -801580.3019,   -6604.9499,      11.6985],
           [ -16696.9582, -801768.0847,   -6604.9499,      12.842 ],
           [ -16520.    , -802110.    ,   -7125.    ,       0.    ]])

    In [15]: b.ox[:,0]
    Out[15]: 
    A()sliced
    A([[ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           ..., 
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ],
           [ -16697.0586, -801768.0625,   -6605.    ,      12.842 ]], dtype=float32)

    In [17]: a.ox[:7500,0] - b.ox[:,0]
    Out[17]: 
    A()sliced
    A([[-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           ..., 
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948],
           [-129.3359,  192.6875,    0.    ,   -1.4948]], dtype=float32)


Termination boundaries
------------------------

::

    134 #define FLAGS(p, s, prd) \
    135 { \
    136     p.flags.i.x = prd.boundary ;  \
    137     p.flags.u.y = s.identity.w ;  \
    138     p.flags.u.w |= s.flag ; \
    139 } \


::

    ( 37) om:               MineralOil os:             RSOilSurface is:                          im:                  Acrylic

    (signed boundaries are 1-based, as 0 means miss : so subtract 1 for the 0-based op --bnd)

    GSurLib::pushBorderSurfaces does not list it, so it should be isur/osur duped in order to be relevant in both directions ???

    WHAT IS THE CG4 8? just the slot 

    HUH : -ve boundary corresponds to inward going photons  ???


    In [21]: a.ox[:,3].view(np.int32)
    Out[21]: 
    A()sliced
    A([[     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272],
           ..., 
           [     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272],
           [     -38,        0, 67305984,     6272]], dtype=int32)

    In [22]: b.ox[:,3].view(np.int32)
    Out[22]: 
    A()sliced
    A([[       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272],
           ..., 
           [       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272],
           [       8,        0, 67305984,     6272]], dtype=int32)


::

    586 void CRecorder::RecordPhoton(const G4Step* step)
    587 {
    588     // gets called at last step (eg absorption) or when truncated
    ...
    609     target->setUInt(target_record_id, 3, 0, 0, m_slot );
    610     target->setUInt(target_record_id, 3, 0, 1, 0u );
    611     target->setUInt(target_record_id, 3, 0, 2, m_c4.u );
    612     target->setUInt(target_record_id, 3, 0, 3, m_mskhis );
    613 


z is c4::

    309     // initial quadrant 
    310     uifchar4 c4 ;
    311     c4.uchar_.x =
    312                   (  p.position.x > 0.f ? QX : 0u )
    313                    |
    314                   (  p.position.y > 0.f ? QY : 0u )
    315                    |
    316                   (  p.position.z > 0.f ? QZ : 0u )
    317                   ;
    318 
    319     c4.uchar_.y = 2u ;   // 3-bytes up for grabs
    320     c4.uchar_.z = 3u ;
    321     c4.uchar_.w = 4u ;
    322 
    323     p.flags.f.z = c4.f ;


    In [28]: a.c4
    Out[28]: 
    rec.array([(0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4), ..., (0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4)], 
          dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1'), ('w', 'u1')])

    In [29]: b.c4
    Out[29]: 
    rec.array([(0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4), ..., (0, 2, 3, 4), (0, 2, 3, 4), (0, 2, 3, 4)], 
          dtype=[('x', 'u1'), ('y', 'u1'), ('z', 'u1'), ('w', 'u1')])




* old groupvel timing issue apparent, fixing that will help with this
* looks like CG4 is taking a few steps more prior to SA



probable cause CG4 logical skin surfaces lacking lv
-----------------------------------------------------

::

    2016-10-02 16:51:37.006 INFO  [1411044] [CBorderSurfaceTable::init@21] CBorderSurfaceTable::init nsurf 11
        0               NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead #0 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner #0
        1                NearOWSLinerSurface pv1 /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS #0 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner #0
        2              NearIWSCurtainSurface pv1 /dd/Geometry/Pool/lvNearPoolCurtain#pvNearPoolIWS #0 pv2 /dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain #0
        3               SSTWaterSurfaceNear1 pv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE1 #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0
        4                      SSTOilSurface pv1 /dd/Geometry/AD/lvSST#pvOIL #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0
        5                      SSTOilSurface pv1 /dd/Geometry/AD/lvSST#pvOIL #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0
        6                   ESRAirSurfaceTop pv1 /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap #0 pv2 /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR #0
        7                   ESRAirSurfaceTop pv1 /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap #0 pv2 /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR #0
        8                   ESRAirSurfaceBot pv1 /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap #0 pv2 /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR #0
        9                   ESRAirSurfaceBot pv1 /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap #0 pv2 /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR #0
       10               SSTWaterSurfaceNear2 pv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE2 #0 pv2 /dd/Geometry/AD/lvADE#pvSST #0

    2016-10-02 16:51:37.006 INFO  [1411044] [CBorderSurfaceTable::dump@47] CGeometryTest CBorderSurfaceTable
    2016-10-02 16:51:37.006 INFO  [1411044] [CSkinSurfaceTable::init@22] CSkinSurfaceTable::init nsurf 36
        0               NearPoolCoverSurface lv NULL
        1      lvPmtHemiCathodeSensorSurface lv NULL
        2    lvHeadonPmtCathodeSensorSurface lv NULL
        3                       RSOilSurface lv NULL
        4                 AdCableTraySurface lv NULL
        5                PmtMtTopRingSurface lv NULL
        6               PmtMtBaseRingSurface lv NULL
        7                   PmtMtRib1Surface lv NULL
        8                   PmtMtRib2Surface lv NULL
        9                   PmtMtRib3Surface lv NULL
       10                 LegInIWSTubSurface lv NULL
       11                  TablePanelSurface lv NULL
       12                 SupportRib1Surface lv NULL
       13                 SupportRib5Surface lv NULL
       14                   SlopeRib1Surface lv NULL
       15                   SlopeRib5Surface lv NULL
       16            ADVertiCableTraySurface lv NULL
       17           ShortParCableTraySurface lv NULL
       18              NearInnInPiperSurface lv NULL
       19             NearInnOutPiperSurface lv NULL
       20                 LegInOWSTubSurface lv NULL
       21                UnistrutRib6Surface lv NULL
       22                UnistrutRib7Surface lv NULL
       23                UnistrutRib3Surface lv NULL
       24                UnistrutRib5Surface lv NULL
       25                UnistrutRib4Surface lv NULL
       26                UnistrutRib1Surface lv NULL
       27                UnistrutRib2Surface lv NULL
       28                UnistrutRib8Surface lv NULL
       29                UnistrutRib9Surface lv NULL
       30           TopShortCableTraySurface lv NULL
       31          TopCornerCableTraySurface lv NULL
       32              VertiCableTraySurface lv NULL
       33              NearOutInPiperSurface lv NULL
       34             NearOutOutPiperSurface lv NULL
       35                LegInDeadTubSurface lv NULL







full seq following fixed aim
--------------------------------

::

      A:seqhis_ana      1:laser 
              8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                  4d        0.055            553       [2 ] TO AB
          cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012            122       [4 ] TO BT BT AB
             8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                 45d        0.006             65       [3 ] TO RE AB
              4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
            8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                455d        0.003             34       [4 ] TO RE RE AB
          cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
             8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
           8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
               4cccd        0.003             25       [5 ] TO BT BT BT AB
          cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                 46d        0.002             21       [3 ] TO SC AB
          cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
            4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                           10000         1.00 
       B:seqhis_ana     -1:laser 
            8ccccccd        0.750           7500       [8 ] TO BT BT BT BT BT BT SA
                  4d        0.074            741       [2 ] TO AB
          cc9ccccccd        0.043            433       [10] TO BT BT BT BT BT BT DR BT BT
          cb9ccccccd        0.027            271       [10] TO BT BT BT BT BT BT DR BR BT
                4ccd        0.018            175       [4 ] TO BT BT AB
           8cccccc6d        0.014            138       [9 ] TO SC BT BT BT BT BT BT SA
              4ccccd        0.009             88       [6 ] TO BT BT BT BT AB
          4c9ccccccd        0.008             78       [10] TO BT BT BT BT BT BT DR BT AB
            4ccccccd        0.007             70       [8 ] TO BT BT BT BT BT BT AB
          cacccccc6d        0.004             35       [10] TO SC BT BT BT BT BT BT SR BT
           8cc6ccccd        0.003             25       [9 ] TO BT BT BT BT SC BT BT SA
          cccc6ccccd        0.002             22       [10] TO BT BT BT BT SC BT BT BT BT
           89ccccccd        0.002             22       [9 ] TO BT BT BT BT BT BT DR SA
          ccbccccc6d        0.002             22       [10] TO SC BT BT BT BT BT BR BT BT
               4cccd        0.002             21       [5 ] TO BT BT BT AB
           8cccc6ccd        0.002             21       [9 ] TO BT BT SC BT BT BT BT SA
           cac0ccc6d        0.002             21       [9 ] TO SC BT BT BT ?0? BT SR BT
                 46d        0.002             18       [3 ] TO SC AB
          cccccc6ccd        0.002             17       [10] TO BT BT SC BT BT BT BT BT BT
          bc9ccccccd        0.002             16       [10] TO BT BT BT BT BT BT DR BT BR
                           10000         1.00 








initial ana 
-------------

::

    ipython -i $(which tokg4.py) -- --det laser

    /Users/blyth/opticks/ana/tokg4.py --det laser
    writing opticks environment to /tmp/blyth/opticks/opticks_env.bash 
    [2016-10-02 11:10:22,331] p22488 {/Users/blyth/opticks/ana/tokg4.py:25} INFO - tag 1 src torch det laser c2max 2.0  
    [2016-10-02 11:10:22,397] p22488 {/Users/blyth/opticks/ana/tokg4.py:36} INFO -  a : laser/torch/  1 :  20161002-1106 /tmp/blyth/opticks/evt/laser/torch/1/fdom.npy 
    [2016-10-02 11:10:22,397] p22488 {/Users/blyth/opticks/ana/tokg4.py:37} INFO -  b : laser/torch/ -1 :  20161002-1106 /tmp/blyth/opticks/evt/laser/torch/-1/fdom.npy 
    A Evt(  1,"torch","laser","laser/torch/  1 : ", seqs="[]") 20161002-1106 /tmp/blyth/opticks/evt/laser/torch/1
    B Evt( -1,"torch","laser","laser/torch/ -1 : ", seqs="[]") 20161002-1106 /tmp/blyth/opticks/evt/laser/torch/-1
           A:seqhis_ana      1:laser 
                  8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                      4d        0.055            553       [2 ] TO AB
              cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
                 8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                    4ccd        0.012            122       [4 ] TO BT BT AB
                 8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                     45d        0.006             65       [3 ] TO RE AB
                  4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
                8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
                 8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                    455d        0.003             34       [4 ] TO RE RE AB
              cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
                 8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
                 86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
               8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
                   4cccd        0.003             25       [5 ] TO BT BT BT AB
              cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                     46d        0.002             21       [3 ] TO SC AB
              cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
                4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                               10000         1.00 
           B:seqhis_ana     -1:laser 
                8c0cc0cd        0.703           7030       [8 ] TO BT ?0? BT BT ?0? BT SA
                      4d        0.090            899       [2 ] TO AB
              4c9c0cc0cd        0.030            301       [10] TO BT ?0? BT BT ?0? BT DR BT AB
              cb9c0cc0cd        0.029            285       [10] TO BT ?0? BT BT ?0? BT DR BR BT
                  4cc0cd        0.022            217       [6 ] TO BT ?0? BT BT AB
                    40cd        0.020            201       [4 ] TO BT ?0? AB
               8cccccc6d        0.015            152       [9 ] TO SC BT BT BT BT BT BT SA
                4c0cc0cd        0.015            145       [8 ] TO BT ?0? BT BT ?0? BT AB
              bb9c0cc0cd        0.011            105       [10] TO BT ?0? BT BT ?0? BT DR BR BR
               cac0ccc6d        0.005             52       [9 ] TO SC BT BT BT ?0? BT SR BT
                     46d        0.005             49       [3 ] TO SC AB
              cc0b0ccc6d        0.004             44       [10] TO SC BT BT BT ?0? BR ?0? BT BT
              cc9c0cc0cd        0.004             43       [10] TO BT ?0? BT BT ?0? BT DR BT BT
              cacccccc6d        0.004             40       [10] TO SC BT BT BT BT BT BT SR BT
              4c6c0cc0cd        0.004             39       [10] TO BT ?0? BT BT ?0? BT SC BT AB
              cccc6cc0cd        0.002             21       [10] TO BT ?0? BT BT SC BT BT BT BT
                     4cd        0.002             20       [3 ] TO BT AB
              bc6c0cc0cd        0.002             17       [10] TO BT ?0? BT BT ?0? BT SC BT BR
              c9cccccc6d        0.002             17       [10] TO SC BT BT BT BT BT BT DR BT
              cccccccc6d        0.002             17       [10] TO SC BT BT BT BT BT BT BT BT
                               10000         1.00 

           A:seqmat_ana      1:laser 
                  443231        0.774           7736       [6 ] Gd Ac LS Ac MO MO
                      11        0.055            553       [2 ] Gd Gd
                 4432311        0.031            314       [7 ] Gd Gd Ac LS Ac MO MO
              3323443231        0.026            265       [10] Gd Ac LS Ac MO MO Ac LS Ac Ac
                    2231        0.012            122       [4 ] Gd Ac LS LS
                     111        0.009             86       [3 ] Gd Gd Gd
                44323111        0.007             72       [8 ] Gd Gd Gd Ac LS Ac MO MO
                 4432231        0.007             71       [7 ] Gd Ac LS LS Ac MO MO
                 4443231        0.005             46       [7 ] Gd Ac LS Ac MO MO MO
              fff3432311        0.004             39       [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ai
              3323132231        0.004             39       [10] Gd Ac LS LS Ac Gd Ac LS Ac Ac
                    1111        0.004             35       [4 ] Gd Gd Gd Gd
              4433432311        0.003             33       [10] Gd Gd Ac LS Ac MO Ac Ac MO MO
               443231111        0.003             31       [9 ] Gd Gd Gd Gd Ac LS Ac MO MO
                aa332311        0.003             26       [8 ] Gd Gd Ac LS Ac Ac ES ES
                   33231        0.003             25       [5 ] Gd Ac LS Ac Ac
                   11111        0.002             20       [5 ] Gd Gd Gd Gd Gd
                dd432311        0.002             20       [8 ] Gd Gd Ac LS Ac MO Vm Vm
                44322231        0.002             17       [8 ] Gd Ac LS LS LS Ac MO MO
                     331        0.001             14       [3 ] Gd Ac Ac
                               10000         1.00 
           B:seqmat_ana     -1:laser 
                44332331        0.718           7175       [8 ] Gd Ac Ac LS Ac Ac MO MO
                      11        0.090            899       [2 ] Gd Gd
              ff44332331        0.034            340       [10] Gd Ac Ac LS Ac Ac MO MO Ai Ai
              3444332331        0.026            264       [10] Gd Ac Ac LS Ac Ac MO MO MO Ac
                  332331        0.022            217       [6 ] Gd Ac Ac LS Ac Ac
                    3331        0.020            201       [4 ] Gd Ac Ac Ac
               443432311        0.015            154       [9 ] Gd Gd Ac LS Ac MO Ac MO MO
              4444332331        0.013            134       [10] Gd Ac Ac LS Ac Ac MO MO MO MO
              33ff332311        0.005             52       [10] Gd Gd Ac LS Ac Ac Ai Ai Ac Ac
              f344332331        0.005             51       [10] Gd Ac Ac LS Ac Ac MO MO Ac Ai
                     111        0.005             49       [3 ] Gd Gd Gd
              3233332311        0.004             43       [10] Gd Gd Ac LS Ac Ac Ac Ac LS Ac
              3ff3432311        0.004             40       [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ac
              3344332331        0.003             29       [10] Gd Ac Ac LS Ac Ac MO MO Ac Ac
              f444332331        0.003             29       [10] Gd Ac Ac LS Ac Ac MO MO MO Ai
                     331        0.002             20       [3 ] Gd Ac Ac
               444332331        0.002             19       [9 ] Gd Ac Ac LS Ac Ac MO MO MO
              3443432311        0.002             17       [10] Gd Gd Ac LS Ac MO Ac MO MO Ac
              3232332331        0.002             16       [10] Gd Ac Ac LS Ac Ac LS Ac LS Ac
              3433432311        0.002             15       [10] Gd Gd Ac LS Ac MO Ac Ac MO Ac
                               10000         1.00 


    






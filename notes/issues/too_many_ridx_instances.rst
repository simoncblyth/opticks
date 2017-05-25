Too Many Ridx Instances
=========================


Initial simple instancing criteria from NScene::labelTree_r of "num_mesh_instances > 4" 
is yielding 117 distinct types of geometry, that is an order of magnitude more than desired.

The very simple instancing criteria based on mesh instance counts alone 
yields many more instances than optimal as it is unaware of the containment relations 
within the node tree. 

Using containment relations allow "vertical" combination in the node tree not just 
the current "horizontal" combinations. 

NScene now does repeat candidate finding similar to GTreeCheck/GNode progeny digests within NScene. 



Can siblings be grouped into added subtrees in crowded nodes ?
----------------------------------------------------------------

* repeat candidate finding only works for subtrees (not siblings), so would be better
  to unflatten the node tree for crowded nodes into groups, allowing the progeny digest 
  and repeat candidiate finding/labelling machinery to be used as is ... 

* eg would want the PMT Collar to be grouped together with the PMT subtree, but 
  its unscalable to write geometry specific code to do this : need a general way 


Where to implement sibling grouping ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* needs only the mesh_id and transforms for the node tree, so could 
  be done at python level 

* seems that it is possible to duplicate transform digests (for identity transforms anyhow) 
  in python (see sysrap/tests/SDigestTest.py) ... hmm but dont want 
  to duplicate the polar decomposition py side, but dont need to : just adjust 
  to digesting the basis transform 
   

Finding nodes with lots of children
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::


    simon:opticks blyth$ tgltf-;tgltf-gdml-rip   # starting from 3153

    In [20]: map(lambda n:(n,len(target.find_nodes_nchild(n))), range(23) )
    Out[20]: 
    [(0, 532),
     (1, 269),
     (2, 228),
     (3, 29),
     (4, 25),
     (5, 13),
     (6, 6),
     (7, 6),
     (8, 6),
     (9, 6),
     (10, 6),
     (11, 2),
     (12, 2),
     (13, 2),
     (14, 2),
     (15, 2),
     (16, 2),
     (17, 2),
     (18, 2),
     (19, 2),
     (20, 2),
     (21, 2),
     (22, 2)]


    In [21]: nn = target.find_nodes_nchild(22)

    In [24]: len(nn[0].children)
    Out[24]: 520

    In [25]: len(nn[1].children)
    Out[25]: 35

    In [29]: nn[0].lv.name
    Out[29]: '/dd/Geometry/AD/lvOIL0xbf5e0b8'

    In [30]: nn[1].lv.name
    Out[30]: '/dd/Geometry/AD/lvLSO0xc403e40'

    In [45]: txf = [c.pv.transform for c in oil.children]

    In [46]: len(oil.children)
    Out[46]: 520

    In [47]: len(txf)
    Out[47]: 520

    In [52]: tt = np.vstack(txf).reshape(-1,4,4)

    In [53]: len(tt)
    Out[53]: 520

    In [54]: tt[0]
    Out[54]: 
    array([[ -1.,   0.,   0.,   0.],
           [ -0.,  -1.,   0.,   0.],
           [  0.,   0.,   1.,   0.],
           [  0.,   0., -49.,   1.]], dtype=float32)

    In [55]: tt[1]
    Out[55]: 
    array([[    0.    ,    -0.    ,     1.    ,     0.    ],
           [    0.1305,    -0.9914,    -0.    ,     0.    ],
           [    0.9914,     0.1305,     0.    ,     0.    ],
           [-2304.6135,  -303.4081, -1750.    ,     1.    ]], dtype=float32)

    In [56]: tt[2]
    Out[56]: 
    array([[    0.    ,    -0.    ,     1.    ,     0.    ],
           [    0.1305,    -0.9914,    -0.    ,     0.    ],
           [    0.9914,     0.1305,     0.    ,     0.    ],
           [-2249.0928,  -296.0987, -1750.    ,     1.    ]], dtype=float32)

    In [57]: tt[3]
    Out[57]: 
    array([[    0.    ,    -0.    ,     1.    ,     0.    ],
           [    0.3827,    -0.9239,    -0.    ,     0.    ],
           [    0.9239,     0.3827,     0.    ,     0.    ],
           [-2147.5579,  -889.5477, -1750.    ,     1.    ]], dtype=float32)

    In [58]: tt[4]
    Out[58]: 
    array([[    0.    ,    -0.    ,     1.    ,     0.    ],
           [    0.3827,    -0.9239,    -0.    ,     0.    ],
           [    0.9239,     0.3827,     0.    ,     0.    ],
           [-2095.8208,  -868.1174, -1750.    ,     1.    ]], dtype=float32)


    In [64]: len(unique2D_subarray(tt))    ## all transforms are unique
    Out[64]: 520



    In [71]: rr = tt[:,:3,:3]

    In [72]: rr
    Out[72]: 
    array([[[-1.    ,  0.    ,  0.    ],
            [-0.    , -1.    ,  0.    ],
            [ 0.    ,  0.    ,  1.    ]],

           [[ 0.    , -0.    ,  1.    ],
            [ 0.1305, -0.9914, -0.    ],
            [ 0.9914,  0.1305,  0.    ]],

           [[ 0.    , -0.    ,  1.    ],
            [ 0.1305, -0.9914, -0.    ],
            [ 0.9914,  0.1305,  0.    ]],

           ..., 
           [[ 0.    , -0.    , -1.    ],
            [ 0.866 , -0.5   ,  0.    ],
            [-0.5   , -0.866 ,  0.    ]],

           [[ 0.    , -0.    , -1.    ],
            [ 0.866 , -0.5   ,  0.    ],
            [-0.5   , -0.866 ,  0.    ]],

           [[ 0.    , -0.    , -1.    ],
            [ 0.866 , -0.5   ,  0.    ],
            [-0.5   , -0.866 ,  0.    ]]], dtype=float32)

    In [73]: len(rr)
    Out[73]: 520

    In [74]: unique2D_subarray(rr)
    Out[74]: 
    array([[[ 1.    ,  0.    ,  0.    ],
            [ 0.    ,  1.    ,  0.    ],
            [ 0.    ,  0.    ,  1.    ]],

           [[-1.    ,  0.    ,  0.    ],
            [-0.    , -1.    ,  0.    ],
            [ 0.    ,  0.    ,  1.    ]],

           [[ 0.    ,  0.    ,  1.    ],
            ...

    In [75]: len(unique2D_subarray(rr))   ## only 68 distinct rotations
    Out[75]: 68




General Sibling grouping 
---------------------------

For crowded nodes like oil and ls, counting 
lv occurrence and looking for groups with equal counts
will yield candidate groupings... then need to analyse the 
transforms to pair the appropriate ones together.  They should
have equal transforms (or at least equal rotation and z-shift) 

Also need way to verify that the intended added groups would actually 
have same progeny digests before going to trouble of editing 
the node tree...



::

    In [1]: tree.analyse_crowds()
    /dd/Geometry/RPC/lvNearRPCRoof0xbf40030 54
        54 : /dd/Geometry/RPC/lvRPCMod0xbf54e60 
    /dd/Geometry/RPCSupport/lvNearHbeamSmallUnit0xc5bef70 72
         2 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSpanHbeam0xc21f438 
         2 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSideShortHbeam0xc2b1dd0 
         8 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagSquareIron0xc358910 
         8 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagAngleIron0xc12bb90 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearPentagonIron0xc35a0b0 
        18 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSquareIron0xc2484c0 
        18 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartLongAngleIron0xc21e000 
    /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a988 178
         2 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSideLongHbeam0xbf3b550 
         4 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSpanHbeam0xc21f438 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagSquareIron0xc358910 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagAngleIron0xc12bb90 
        18 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartShortAngleIron0xbf3dbf0 
        32 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearPentagonIron0xc35a0b0 
        36 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartLongAngleIron0xc21e000 
        54 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSquareIron0xc2484c0 
    /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a988 178
         2 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSideLongHbeam0xbf3b550 
         4 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSpanHbeam0xc21f438 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagSquareIron0xc358910 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagAngleIron0xc12bb90 
        18 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartShortAngleIron0xbf3dbf0 
        32 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearPentagonIron0xc35a0b0 
        36 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartLongAngleIron0xc21e000 
        54 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSquareIron0xc2484c0 
    /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a988 178
         2 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSideLongHbeam0xbf3b550 
         4 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSpanHbeam0xc21f438 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagSquareIron0xc358910 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagAngleIron0xc12bb90 
        18 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartShortAngleIron0xbf3dbf0 
        32 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearPentagonIron0xc35a0b0 
        36 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartLongAngleIron0xc21e000 
        54 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSquareIron0xc2484c0 
    /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a988 178
         2 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSideLongHbeam0xbf3b550 
         4 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSpanHbeam0xc21f438 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagSquareIron0xc358910 
        16 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearDiagAngleIron0xc12bb90 
        18 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartShortAngleIron0xbf3dbf0 
        32 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearPentagonIron0xc35a0b0 
        36 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearThwartLongAngleIron0xc21e000 
        54 : /dd/Geometry/RPCSupport/TrivialComponents/lvNearSquareIron0xc2484c0 
    /dd/Geometry/Pool/lvNearPoolOWS0xbf93840 2938
         1 : /dd/Geometry/Pool/lvNearPoolCurtain0xc2ceef0 
         1 : /dd/Geometry/PoolDetails/lvOutInWaterPipeNearTub0xce594c0 
         1 : /dd/Geometry/PoolDetails/lvOutOutWaterPipeNearTub0xce58ca0 
         2 : /dd/Geometry/PoolDetails/lvTopShortCableTray0xce58200 
         4 : /dd/Geometry/PoolDetails/lvTopCornerCableTray0xce56ff8 
         8 : /dd/Geometry/PoolDetails/lvLegInOWSTub0xcced348 
         8 : /dd/Geometry/PoolDetails/lvVertiCableTray0xc0e08a0 
        16 : /dd/Geometry/PoolDetails/lvShortParRib20xcd56b40 
        16 : /dd/Geometry/PoolDetails/lvLongParRib20xc3b4910 
        16 : /dd/Geometry/PoolDetails/lvShortParRib10xcd55e48 
        16 : /dd/Geometry/PoolDetails/lvLongParRib10xc3b3eb8 
        32 : /dd/Geometry/PoolDetails/lvCornerParRib10xc0e2430 
        32 : /dd/Geometry/PoolDetails/lvCornerParRib20xc0f2040 
        92 : /dd/Geometry/PoolDetails/lvBotVertiRib0xbf63800 

       167 : /dd/Geometry/PMT/lvPmtTee0xc011648 
       167 : /dd/Geometry/PMT/lvPmtHemi0xc133740 
       167 : /dd/Geometry/PMT/lvPmtTopRing0xc3486f0 
       167 : /dd/Geometry/PMT/lvPmtBaseRing0xc00f400        ### this would be a juicy instance

       192 : /dd/Geometry/PoolDetails/lvCrossRib0xcd570b8 

       330 : /dd/Geometry/PoolDetails/lvSidVertiRib0xc5e6fa0 

       501 : /dd/Geometry/PMT/lvMountRib10xc3a4cb0 
       501 : /dd/Geometry/PMT/lvMountRib20xc012500          
       501 : /dd/Geometry/PMT/lvMountRib30xc00d350          ### maybe this too 


    /dd/Geometry/Pool/lvNearPoolIWS0xc28bc60 1619
         1 : /dd/Geometry/PoolDetails/lvInnInWaterPipeNearTub0xbf29660 
         1 : /dd/Geometry/PoolDetails/lvInnOutWaterPipeNearTub0xc0d7c30 
         2 : /dd/Geometry/PoolDetails/lvInnShortParCableTray0xc95a730 
         2 : /dd/Geometry/AD/lvADE0xc2a78c0 
         2 : /dd/Geometry/PoolDetails/lvTablePanel0xc0101d8 
         2 : /dd/Geometry/PoolDetails/lvInnVertiCableTray0xbf28e40 
         4 : /dd/Geometry/PoolDetails/lvSupportRib50xc0d8bb8 
         8 : /dd/Geometry/PoolDetails/lvLegInIWSTub0xc400e40 
         8 : /dd/Geometry/PoolDetails/lvSlopeRib10xc0d8b50 
         8 : /dd/Geometry/PoolDetails/lvSupportRib10xc0d8868 
         8 : /dd/Geometry/PoolDetails/lvSlopeRib50xc0d8db0 

       121 : /dd/Geometry/PMT/lvPmtTee0xc011648 
       121 : /dd/Geometry/PMT/lvPmtTopRing0xc3486f0 
       121 : /dd/Geometry/PMT/lvPmtBaseRing0xc00f400 
       121 : /dd/Geometry/PMT/lvPmtHemi0xc133740          ### another juicy one if can be grouped into identical mesh-transform-digest subtrees 

       363 : /dd/Geometry/PMT/lvMountRib20xc012500 
       363 : /dd/Geometry/PMT/lvMountRib30xc00d350 
       363 : /dd/Geometry/PMT/lvMountRib10xc3a4cb0         ### perhaps

    /dd/Geometry/AD/lvOIL0xbf5e0b8 520
         1 : /dd/Geometry/AdDetails/lvSstTopHub0xc2644f0 
         1 : /dd/Geometry/AdDetails/lvOcrGdsLsoPrt0xc104a90 
         1 : /dd/Geometry/AdDetails/lvOcrCalLsoPrt0xc1077c8 
         1 : /dd/Geometry/AdDetails/lvSstBotHub0xc1760f0 
         1 : /dd/Geometry/AdDetails/lvTopReflector0xbf9be68 
         1 : /dd/Geometry/AdDetails/lvCtrLsoOflInOil0xc183248 
         1 : /dd/Geometry/AdDetails/lvOcrGdsLsoOfl0xc1052d0 
         1 : /dd/Geometry/AdDetails/lvOcrCalLso0xc17e288 
         1 : /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0 
         1 : /dd/Geometry/AD/lvOAV0xbf1c760 
         1 : /dd/Geometry/AdDetails/lvOavTopHub0xbf366d0 
         3 : /dd/Geometry/AdDetails/lvCtrLsoOflTopClp0xc26f5a0 
         3 : /dd/Geometry/CalibrationSources/lvWallLedSourceAssy0xc3a9f40 
         3 : /dd/Geometry/AdDetails/lvCtrLsoOflTfb0xc3a2ab0 
         4 : /dd/Geometry/AdDetails/lvBotRefRadialShortRib0xbf339c8 
         4 : /dd/Geometry/AdDetails/lvBotRefRadialLongRib0xbf32988 
         6 : /dd/Geometry/PMT/lvHeadonPmtAssy0xbf9fb20 
         6 : /dd/Geometry/PMT/lvHeadonPmtMount0xc02d380 
         8 : /dd/Geometry/AdDetails/lvSstInnVerRibBase0xbf31748 
         8 : /dd/Geometry/AdDetails/lvSstBotRib0xc26c650 
         8 : /dd/Geometry/AdDetails/lvSstTopTshapeRib0xc2629f0 
         8 : /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0 
         8 : /dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220 
         8 : /dd/Geometry/AdDetails/lvOavTopRib0xbf7bce8 
         8 : /dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0 
         8 : /dd/Geometry/AdDetails/lvBotRefCircleRib0xbf34468 
        32 : /dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0 

       192 : /dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0 
       192 : /dd/Geometry/PMT/lvPmtHemi0xc133740           ### obvious one


    /dd/Geometry/AD/lvLSO0xc403e40 35

         1 : /dd/Geometry/AD/lvIAV0xc404ee8 
         1 : /dd/Geometry/AdDetails/lvIavTopHub0xc129d88 
         1 : /dd/Geometry/AdDetails/lvCtrGdsOflInLso0xc28cc88 
         1 : /dd/Geometry/AdDetails/lvIavBotHub0xc355b80 
         1 : /dd/Geometry/AdDetails/lvCtrGdsOflTfbInLso0xbfa0728 
         1 : /dd/Geometry/AdDetails/lvOcrGdsPrt0xc352630 
         1 : /dd/Geometry/AdDetails/lvOavBotHub0xc3550d8 
         1 : /dd/Geometry/AdDetails/lvOcrGdsTfbInLso0xc3529c0 
         1 : /dd/Geometry/AdDetails/lvOcrGdsInLso0xc353990 
         2 : /dd/Geometry/AdDetails/lvCtrGdsOflBotClp0xc407eb0 
         8 : /dd/Geometry/AdDetails/lvIavBotRib0xc355990 
         8 : /dd/Geometry/AdDetails/lvOavBotRib0xc353d30 
         8 : /dd/Geometry/AdDetails/lvIavTopRib0xbf8e280 

    /dd/Geometry/AD/lvOIL0xbf5e0b8 520
         1 : /dd/Geometry/AdDetails/lvSstTopHub0xc2644f0 
         1 : /dd/Geometry/AdDetails/lvOcrGdsLsoPrt0xc104a90 
         1 : /dd/Geometry/AdDetails/lvOcrCalLsoPrt0xc1077c8 
         1 : /dd/Geometry/AdDetails/lvSstBotHub0xc1760f0 
         1 : /dd/Geometry/AdDetails/lvTopReflector0xbf9be68 
         1 : /dd/Geometry/AdDetails/lvCtrLsoOflInOil0xc183248 
         1 : /dd/Geometry/AdDetails/lvOcrGdsLsoOfl0xc1052d0 
         1 : /dd/Geometry/AdDetails/lvOcrCalLso0xc17e288 
         1 : /dd/Geometry/AdDetails/lvBotReflector0xc3cd4c0 
         1 : /dd/Geometry/AD/lvOAV0xbf1c760 
         1 : /dd/Geometry/AdDetails/lvOavTopHub0xbf366d0 
         3 : /dd/Geometry/AdDetails/lvCtrLsoOflTopClp0xc26f5a0 
         3 : /dd/Geometry/CalibrationSources/lvWallLedSourceAssy0xc3a9f40 
         3 : /dd/Geometry/AdDetails/lvCtrLsoOflTfb0xc3a2ab0 
         4 : /dd/Geometry/AdDetails/lvBotRefRadialShortRib0xbf339c8 
         4 : /dd/Geometry/AdDetails/lvBotRefRadialLongRib0xbf32988 
         6 : /dd/Geometry/PMT/lvHeadonPmtAssy0xbf9fb20 
         6 : /dd/Geometry/PMT/lvHeadonPmtMount0xc02d380 
         8 : /dd/Geometry/AdDetails/lvSstInnVerRibBase0xbf31748 
         8 : /dd/Geometry/AdDetails/lvSstBotRib0xc26c650 
         8 : /dd/Geometry/AdDetails/lvSstTopTshapeRib0xc2629f0 
         8 : /dd/Geometry/AdDetails/lvSstTopRadiusRib0xc2716c0 
         8 : /dd/Geometry/AdDetails/lvSstBotCirRibBase0xc26e220 
         8 : /dd/Geometry/AdDetails/lvOavTopRib0xbf7bce8 
         8 : /dd/Geometry/AdDetails/lvSstTopCirRibBase0xc2649f0 
         8 : /dd/Geometry/AdDetails/lvBotRefCircleRib0xbf34468 
        32 : /dd/Geometry/AdDetails/lvRadialShieldUnit0xc3d7ec0 

       192 : /dd/Geometry/PMT/lvAdPmtCollar0xbf21fb0           ### from the other AD
       192 : /dd/Geometry/PMT/lvPmtHemi0xc133740 


    /dd/Geometry/AD/lvLSO0xc403e40 35
         1 : /dd/Geometry/AD/lvIAV0xc404ee8 
         1 : /dd/Geometry/AdDetails/lvIavTopHub0xc129d88 
         1 : /dd/Geometry/AdDetails/lvCtrGdsOflInLso0xc28cc88 
         1 : /dd/Geometry/AdDetails/lvIavBotHub0xc355b80 
         1 : /dd/Geometry/AdDetails/lvCtrGdsOflTfbInLso0xbfa0728 
         1 : /dd/Geometry/AdDetails/lvOcrGdsPrt0xc352630 
         1 : /dd/Geometry/AdDetails/lvOavBotHub0xc3550d8 
         1 : /dd/Geometry/AdDetails/lvOcrGdsTfbInLso0xc3529c0 
         1 : /dd/Geometry/AdDetails/lvOcrGdsInLso0xc353990 
         2 : /dd/Geometry/AdDetails/lvCtrGdsOflBotClp0xc407eb0 
         8 : /dd/Geometry/AdDetails/lvIavBotRib0xc355990 
         8 : /dd/Geometry/AdDetails/lvOavBotRib0xc353d30 
         8 : /dd/Geometry/AdDetails/lvIavTopRib0xbf8e280 





Initial Approach
-------------------

::

    252 unsigned NScene::deviseRepeatIndex(nd* n)
    253 {
    254     unsigned mesh_idx = n->mesh ;
    255     unsigned num_mesh_instances = getNumInstances(mesh_idx) ;
    256 
    257     unsigned ridx = 0 ;   // <-- global default ridx
    258 
    259     bool make_instance  = num_mesh_instances > 4  ;
    260 
    261     if(make_instance)
    262     {
    263         if(m_mesh2ridx.count(mesh_idx) == 0)
    264              m_mesh2ridx[mesh_idx] = m_mesh2ridx.size() + 1 ;
    265 
    266         ridx = m_mesh2ridx[mesh_idx] ;
    267 
    268         // ridx is a 1-based contiguous index tied to the mesh_idx 
    269         // using trivial things like "mesh_idx + 1" causes  
    270         // issue downstream which expects a contiguous range of ridx 
    271         // when using partial geometries 
    272     }
    273     return ridx ;
    274 }
    275 
    276 void NScene::labelTree_r(nd* n)
    277 {
    278     unsigned ridx = deviseRepeatIndex(n);
    279 
    280     n->repeatIdx = ridx ;
    281 
    282     if(m_repeat_count.count(ridx) == 0) m_repeat_count[ridx] = 0 ;
    283     m_repeat_count[ridx]++ ;
    284 
    285 
    286     for(nd* c : n->children) labelTree_r(c) ;
    287 }





::

    tgltf-;tgltf-gdml

    2017-05-24 12:22:23.820 INFO  [2974756] [*GScene::createVolumeTree@131] GScene::createVolumeTree DONE num_nodes: 12229
    2017-05-24 12:22:23.851 INFO  [2974756] [GScene::makeMergedMeshAndInstancedBuffers@269] GScene::makeMergedMeshAndInstancedBuffers num_repeats 117 START 
    2017-05-24 12:22:54.614 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 1
    2017-05-24 12:22:54.683 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 1
    2017-05-24 12:22:55.255 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 11
    2017-05-24 12:22:55.334 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 15
    2017-05-24 12:22:55.483 WARN  [2974756] [GMesh::allocate@614] GMesh::allocate EMPTY numVertices 0 numFaces 0 numSolids 33
    2017-05-24 12:22:56.197 INFO  [2974756] [GScene::makeMergedMeshAndInstancedBuffers@319] GScene::makeMergedMeshAndInstancedBuffers DONE num_repeats 117 nmm_created 117 nmm 117
    Assertion failed: (0 && "early exit for gltf==4"), function loadFromGLTF, file /Users/blyth/opticks/ggeo/GGeo.cc, line 660.




GTreeCheck triangulated approach
-----------------------------------------


::

    027 GTreeCheck::GTreeCheck(GGeo* ggeo)
     28        :
     29        m_ggeo(ggeo),
     30        m_geolib(ggeo->getGeoLib()),
     31        m_repeat_min(120),
     32        m_vertex_min(300),   // aiming to include leaf? sStrut and sFasteners
     33        m_root(NULL),
     34        m_count(0),
     35        m_labels(0),
     36        m_digest_count(new Counts<unsigned>("progenyDigest"))
     37 {
     38 }


     87 void GTreeCheck::traverse()
     88 {
     89     m_root = m_ggeo->getSolid(0);
     90     assert(m_root);
     91 
     92     // count occurences of distinct progeny digests (relative sub-tree identities) in m_digest_count 
     93     traverse_r(m_root, 0);
     94 
     95     m_digest_count->sort(false);   // descending count order, ie most common subtrees first
     96     //m_digest_count->dump();
     97 
     98     // minrep 120 removes repeats from headonPMT, calibration sources and RPC leaving just PMTs 
     99 
    100     // collect digests of repeated pieces of geometry into  m_repeat_candidates
    101     findRepeatCandidates(m_repeat_min, m_vertex_min);
    102     dumpRepeatCandidates();
    103 }
    104 
    105 void GTreeCheck::traverse_r( GNode* node, unsigned int depth)
    106 {
    107     std::string& pdig = node->getProgenyDigest();
    108     m_digest_count->add(pdig.c_str());
    109     m_count++ ;
    110 
    111     for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1 );
    112 }


    155 void GTreeCheck::findRepeatCandidates(unsigned int repeat_min, unsigned int vertex_min)
    156 {
    157     unsigned int nall = m_digest_count->size() ;
    ...
    166     // over distinct subtrees (ie progeny digests)
    167     for(unsigned int i=0 ; i < nall ; i++)
    168     {
    169         std::pair<std::string,unsigned int>&  kv = m_digest_count->get(i) ;
    170 
    171         std::string& pdig = kv.first ;
    172         unsigned int ndig = kv.second ;                 // number of occurences of the progeny digest 
    173 
    174         GNode* node = m_root->findProgenyDigest(pdig) ; // first node that matches the progeny digest
    175 
    176         // suspect problem with allowing leaf repeaters is that digesta are not-specific enough, 
    177         // so get bad matching 
    178         //
    179         //  allowing leaf repeaters results in too many, so place vertex count reqirement too 
    180 
    181 
    182         unsigned int nprog = node->getProgenyCount() ;  // includes self when GNode.m_selfdigest is true
    183         unsigned int nvert = node->getProgenyNumVertices() ;  // includes self when GNode.m_selfdigest is true
    184 
    185        // hmm: maybe selecting based on  ndig*nvert 
    186        // but need to also require ndig > smth as dont want to repeat things like the world 
    187 
    188         bool select = ndig > repeat_min && nvert > vertex_min ;
    189 
    190         if(i < 15) LOG(info)
    191                   << ( select ? "**" : "  " )
    192                   << " i "     << std::setw(3) << i
    193                   << " pdig "  << std::setw(32) << pdig
    194                   << " ndig "  << std::setw(6) << ndig
    195                   << " nprog " <<  std::setw(6) << nprog
    196                   << " nvert " <<  std::setw(6) << nvert
    197                   << " n "     <<  node->getName()
    198                   ;
    199 
    200         if(select) m_repeat_candidates.push_back(pdig);
    201     }
    202 
    203     // erase repeats that are enclosed within other repeats 
    204     // ie that have an ancestor which is also a repeat candidate
    205 
    206     m_repeat_candidates.erase(
    207          std::remove_if(m_repeat_candidates.begin(), m_repeat_candidates.end(), *this ),
    208          m_repeat_candidates.end()
    209     );
    210 
    211 
    212 }
    213 
    214 bool GTreeCheck::operator()(const std::string& dig)
    215 {
    216     bool cr = isContainedRepeat(dig, 3);
    217 
    218     if(cr) LOG(info)
    219                   << "GTreeCheck::operator() "
    220                   << " pdig "  << std::setw(32) << dig
    221                   << " disallowd as isContainedRepeat "
    222                   ;
    223 
    224     return cr ;
    225 }
    226 
    227 bool GTreeCheck::isContainedRepeat( const std::string& pdig, unsigned int levels ) const
    228 {
    229     // for the first node that matches the *pdig* progeny digest
    230     // look back *levels* ancestors to see if any of the immediate ancestors 
    231     // are also repeat candidates, if they are then this is a contained repeat
    232     // and is thus disallowed in favor of the ancestor that contains it 
    233 
    234     GNode* node = m_root->findProgenyDigest(pdig) ;
    235     std::vector<GNode*>& ancestors = node->getAncestors();
    236     unsigned int asize = ancestors.size();
    237 
    238     for(unsigned int i=0 ; i < std::min(levels, asize) ; i++)
    239     {
    240         GNode* a = ancestors[asize - 1 - i] ;
    241         std::string& adig = a->getProgenyDigest();
    242         if(std::find(m_repeat_candidates.begin(), m_repeat_candidates.end(), adig ) != m_repeat_candidates.end())
    243         {
    244             return true ;
    245         }
    246     }
    247     return false ;
    248 }




    015 class GGEO_API GNode {
    ...
    148   private:
    149       std::string         m_local_digest ;
    150       std::string         m_progeny_digest ;
    151       std::vector<GNode*> m_progeny ;
    152       std::vector<GNode*> m_ancestors ;

    024 GNode::GNode(unsigned int index, GMatrixF* transform, GMesh* mesh)
     25     :
     26     m_selfdigest(true),


    442 std::string& GNode::getProgenyDigest()
    443 {
    444     if(m_progeny_digest.empty())
    445     {
    446         std::vector<GNode*>& progeny = getProgeny();
    447         m_progeny_count = progeny.size();
    448         GNode* extra = m_selfdigest ? this : NULL ;
    449         m_progeny_digest = GNode::localDigest(progeny, extra) ;
    450     }
    451     return m_progeny_digest ;
    452 }

    283 std::vector<GNode*>& GNode::getProgeny()
    284 {
    285     if(m_progeny.size() == 0)
    286     {
    287         // call on children, as wish to avoid collecting self  
    288         for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(m_progeny); 
    289     }
    290     return m_progeny ; 
    291 }
    292 
    293 void GNode::collectProgeny(std::vector<GNode*>& progeny)
    294 {
    295     progeny.push_back(this);
    296     for(unsigned int i = 0; i < getNumChildren(); i++) getChild(i)->collectProgeny(progeny);
    297 }



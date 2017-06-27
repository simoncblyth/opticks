Impinging Volumes
=====================


Plan
-------


* start by testing each node bbox against its parent bbox 

  * within solid uncoincence done in NCSG::postimport, analogous
    place for volume overlap testing would be NScene/GScene ? 

  * start with NScene::postimport

  * testing with: tgltf-t 
  

tgltf-t : Look at gds example
----------------------------------

::

    tgltf-;tgltf-t  ## with OPTICKS_QUERY selection to pick two volumes only, and manual dumping


Comparing gds and parent nd volumes in NScene::postimportmesh find that they have coincident bbox in Z.

* this is highly likely to explain the tachyon behaviour


Whats the appropriate fix ?
----------------------------

* nudging CSG (eg a few epsilon decrease_z2 or increase_z1) 
  would apply to all instances, so that might not be appropriate 

  * need to check if all lv are similarly coincident

* otherwise would need to apply a nudge transform to the node ? 


Are there missing transforms ?
----------------------------------

* TODO: examine full structural transform tree, for node and its parent, to look for bugs

::

    Hmm : is there 2.5mm of z translation missing in the parent (iav) gtransform ?

             -7101.5
             -7100.0


    tgltf-;tgltf-t  ## with OPTICKS_QUERY selection to pick two volumes only, and manual dumping



    2017-06-27 14:32:42.057 INFO  [1429523] [NScene::postimport@384] NScene::postimport numNd 12230
    2017-06-27 14:32:42.057 INFO  [1429523] [NScene::dumpNd@613] NScene::dumpNd nidx 3158 node exists  verbosity 1

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3158:  0: 35:  2:13:   0] bnd:LiquidScintillator///Acrylic   
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   2.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7107.500   1.000 


     mesh_id 35 meshmeta NScene::meshmeta mesh_id  35 lvidx  24 height  2 soname                        iav0xc346f90 lvname      /dd/Geometry/AD/lvIAV0xc404ee8


    2017-06-27 14:32:42.057 INFO  [1429523] [NScene::dumpNd@613] NScene::dumpNd nidx 3159 node exists  verbosity 1

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3159:  0: 36:  0:14:   0] bnd:Acrylic///GdDopedLS   
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   7.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 


     mesh_id 36 meshmeta NScene::meshmeta mesh_id  36 lvidx  22 height  2 soname                        gds0xc28d3f0 lvname      /dd/Geometry/AD/lvGDS0xbf6cbb8




NScene::check_containment checking bbox containment of all node/parent pairs
----------------------------------------------------------------------------------

* 30% of volumes have bbox containment issues, including PMT volumes

  * perhaps a missing transform bug ? 
  * all the mn and mx in (mm) in the below table 
    should be +ve, they are zero with coincidence and -ve with protrusion  
  * TODO: check the instanced are correctly treated here


Are checking containment by comparing the globally transformed axis aligned bbox 
of a node and its parent.

* is there a better way to check containment ? 
* rotational transforms change box dimensions (as bbox stays axis aligned), 
* perhaps should transform into parent frame to make comparison ?


::

     514 void NScene::check_containment_r(nd* node)
     515 {
     516     nd* parent = node->parent ;
     517     if(!parent) parent = node ;   // only root should not have parent
     518 
     519     nbbox  nbb = get_bbox( node->idx ) ;
     520     nbbox  pbb = get_bbox( parent->idx ) ;
     521 
     522     float epsilon = 1e-5 ;
     523 
     524     unsigned errmask = nbb.classify_containment( pbb, epsilon );
     525 
     526     node->containment = errmask ;
     527 
     528     if(errmask) m_containment_err++ ;
     529 
     530     //if(m_verbosity > 2 || ( errmask && m_verbosity > 0))
     531     {
     532         glm::vec3 dmin( nbb.min.x - pbb.min.x,
     533                         nbb.min.y - pbb.min.y,
     534                         nbb.min.z - pbb.min.z );
     535 
     536         glm::vec3 dmax( pbb.max.x - nbb.max.x,
     537                         pbb.max.y - nbb.max.y,
     538                         pbb.max.z - nbb.max.z );



     442 nbbox NScene::calc_bbox(const nd* node, bool global) const
     443 {
     444     unsigned mesh_idx = node->mesh ;
     445 
     446     NCSG* csg = getCSG(mesh_idx);
     447     assert(csg);
     448 
     449     nnode* root = csg->getRoot();
     450     assert(root);
     451 
     452     assert( node->gtransform );
     453     const glm::mat4& node_t  = node->gtransform->t ;
     454 
     455     nbbox bb  = root->bbox();
     456 
     457     nbbox gbb = bb.transform(node_t) ;
     458 
     459     if(m_verbosity > 2)
     460     std::cout
     461         << " get_bbox "
     462         << " verbosity " << m_verbosity
     463         << " mesh_idx "  << mesh_idx
     464         << " root "  << root->tag()
     465         << std::endl
     466         << gpresent("node_t", node_t)
     467         << std::endl
     468         << " bb  " <<  bb.desc() << std::endl
     469         << " gbb " <<  gbb.desc() << std::endl
     470         ;
     471 
     472     return global ? gbb : bb ;
     473 }



::

    2017-06-27 20:45:11.089 INFO  [1538289] [NScene::postimportmesh@420] NScene::postimportmesh numNd 12230 dbgnode 3159 verbosity 1
    2017-06-27 20:45:11.116 INFO  [1538289] [NScene::check_containment@498] NScene::check_containment verbosity 1
    NSc::ccr n      0 p      0 mn(n-p) (      0.000     0.000     0.000) mx(p-n) (      0.000     0.000     0.000) pv                            top err XMIN_CO YMIN_CO ZMIN_CO XMAX_CO YMAX_CO ZMAX_CO 
    NSc::ccr n      1 p      0 mn(n-p) ( 2348910.2501563320.1252372890.000) mx(p-n) ( 2381950.2503167540.0002377110.000) pv               db-rock0xc15d358 err 
    NSc::ccr n      2 p      1 mn(n-p) (  20001.729  7258.312 25000.000) mx(p-n) (  12644.018 16790.562 10000.000) pv lvNearSiteRock#pvNearHallTop0x err 
    NSc::ccr n      3 p      2 mn(n-p) (   6024.635 17878.750     0.000) mx(p-n) (  13382.347  8346.500 14956.000) pv lvNearHallTop#pvNearTopCover0x err ZMIN_CO 
    NSc::ccr n      4 p      2 mn(n-p) (  17966.039 28909.250  2754.903) mx(p-n) (  15508.528 13171.500 12167.097) pv lvNearHallTop#pvNearTeleRpc#pv err 
    NSc::ccr n      5 p      4 mn(n-p) (     55.189    38.312     1.500) mx(p-n) (     52.945    60.562     1.500) pv    lvRPCMod#pvRPCFoam0xbf1a820 err 
    NSc::ccr n      6 p      5 mn(n-p) (      6.899     6.875    20.500) mx(p-n) (      6.899     6.875    48.500) pv lvRPCFoam#pvBarCham14Array#pvB err 
    NSc::ccr n      7 p      6 mn(n-p) (     13.797    13.812     2.000) mx(p-n) (     13.797    13.812     2.000) pv lvRPCBarCham14#pvRPCGasgap140x err 
    NSc::ccr n      8 p      7 mn(n-p) (    973.189     0.000     0.000) mx(p-n) (      0.000  1538.000     0.000) pv lvRPCGasgap14#pvStrip14Array#p err YMIN_CO ZMIN_CO XMAX_CO ZMAX_CO 
    NSc::ccr n      9 p      7 mn(n-p) (    834.162   219.750     0.000) mx(p-n) (    139.027  1318.250     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     10 p      7 mn(n-p) (    695.136   439.438     0.000) mx(p-n) (    278.054  1098.562     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     11 p      7 mn(n-p) (    556.108   659.125     0.000) mx(p-n) (    417.081   878.875     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     12 p      7 mn(n-p) (    417.081   878.875     0.000) mx(p-n) (    556.108   659.125     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     13 p      7 mn(n-p) (    278.054  1098.562     0.000) mx(p-n) (    695.136   439.438     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     14 p      7 mn(n-p) (    139.027  1318.250     0.000) mx(p-n) (    834.162   219.750     0.000) pv lvRPCGasgap14#pvStrip14Array#p err ZMIN_CO ZMAX_CO 
    NSc::ccr n     15 p      7 mn(n-p) (      0.000  1538.000     0.000) mx(p-n) (    973.189     0.000     0.000) pv lvRPCGasgap14#pvStrip14Array#p err XMIN_CO ZMIN_CO YMAX_CO ZMAX_CO 
    NSc::ccr n     16 p      5 mn(n-p) (      6.899     6.875    58.500) mx(p-n) (      6.899     6.875    10.500) pv lvRPCFoam#pvBarCham14Array#pvB err 
    NSc::ccr n     17 p     16 mn(n-p) (     13.797    13.812     2.000) mx(p-n) (     13.797    13.812     2.000) pv lvRPCBarCham14#pvRPCGasgap140x err 
    NSc::ccr n     18 p     17 mn(n-p) (    973.189     0.000     0.000) mx(p-n) (      0.000  1538.000     0.000) pv lvRPCGasgap14#pvStrip14Array#p err YMIN_CO ZMIN_CO XMAX_CO ZMAX_CO 
    ...
    NSc::ccr n   3142 p   2968 mn(n-p) (   6025.996  5863.750    42.000) mx(p-n) (   6148.171  3832.000    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3143 p   2968 mn(n-p) (   5132.042  5358.812    42.000) mx(p-n) (   6968.165  4428.938    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3144 p   2968 mn(n-p) (   4675.837  5417.750    42.000) mx(p-n) (   7424.370  4370.000    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3145 p   2968 mn(n-p) (   1851.244  3537.688    42.000) mx(p-n) (  10322.922  6158.062    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3146 p   2968 mn(n-p) (   1710.129  3099.875    42.000) mx(p-n) (  10464.037  6595.875    42.000) pv lvNearHbeamBigUnit#pvNearRight err 
    NSc::ccr n   3147 p      1 mn(n-p) (  25611.527 24722.188 14700.000) mx(p-n) (  25611.527 24722.188 25000.000) pv lvNearSiteRock#pvNearHallBot0x err 
    NSc::ccr n   3148 p   3147 mn(n-p) (    414.836   414.875   300.000) mx(p-n) (    414.838   414.875     0.000) pv lvNearHallBot#pvNearPoolDead0x err ZMAX_CO 
    NSc::ccr n   3149 p   3148 mn(n-p) (    116.156   116.125    84.000) mx(p-n) (    116.155   116.125     0.000) pv lvNearPoolDead#pvNearPoolLiner err ZMAX_CO 
    NSc::ccr n   3150 p   3149 mn(n-p) (      0.000     0.000     4.000) mx(p-n) (      0.000     0.000     0.000) pv lvNearPoolLiner#pvNearPoolOWS0 err XMIN_CO YMIN_CO XMAX_CO YMAX_CO ZMAX_CO 
    NSc::ccr n   3151 p   3150 mn(n-p) (   1388.324  1388.312  1000.000) mx(p-n) (   1388.325  1388.312     0.000) pv lvNearPoolOWS#pvNearPoolCurtai err ZMAX_CO 
    NSc::ccr n   3152 p   3151 mn(n-p) (      0.000     0.000     4.000) mx(p-n) (      0.000     0.000     0.000) pv lvNearPoolCurtain#pvNearPoolIW err XMIN_CO YMIN_CO XMAX_CO YMAX_CO ZMAX_CO 
    NSc::ccr n   3153 p   3152 mn(n-p) (   1676.879  6536.250  1408.000) mx(p-n) (   4795.783  1715.125  1490.000) pv lvNearPoolIWS#pvNearADE10xc2cf err 
    NSc::ccr n   3154 p   3153 mn(n-p) (    345.697   345.688    10.000) mx(p-n) (    345.698   345.688  1000.000) pv           lvADE#pvSST0xc128d90 err 
    NSc::ccr n   3155 p   3154 mn(n-p) (     16.594    16.625    30.000) mx(p-n) (     16.594    16.625    15.000) pv           lvSST#pvOIL0xc241510 err 
    NSc::ccr n   3156 p   3155 mn(n-p) (    619.492   619.500   460.000) mx(p-n) (    619.492   619.500   400.379) pv           lvOIL#pvOAV0xbf8f638 err 
    NSc::ccr n   3157 p   3156 mn(n-p) (     80.201    80.188    18.000) mx(p-n) (     80.202    80.188     0.000) pv           lvOAV#pvLSO0xbf8e120 err ZMAX_CO 
    NSc::ccr n   3158 p   3157 mn(n-p) (    576.625   576.625   442.000) mx(p-n) (    576.625   576.625   460.182) pv           lvLSO#pvIAV0xc2d0348 err 
    NSc::ccr n   3159 p   3158 mn(n-p) (     20.742    20.750    15.000) mx(p-n) (     20.742    20.750     0.000) pv           lvIAV#pvGDS0xbf6ab00 err ZMAX_CO 
    NSc::ccr n   3160 p   3158 mn(n-p) (   1353.928  1009.250  3129.720) mx(p-n) (   2887.104  3231.750   -44.720) pv   lvIAV#pvOcrGdsInIAV0xbf6b0e0 err ZMAX_OUT 
    NSc::ccr n   3161 p   3157 mn(n-p) (   2533.279  2533.250  3616.439) mx(p-n) (   2533.278  2533.250   349.621) pv     lvLSO#pvIavTopHub0xc34e6e8 err 
    NSc::ccr n   3162 p   3157 mn(n-p) (   2533.279  2533.250  3727.000) mx(p-n) (   2533.278  2533.250   319.621) pv lvLSO#pvCtrGdsOflBotClp0xc2ce2 err 
    NSc::ccr n   3163 p   3157 mn(n-p) (   2695.758  2695.750  3757.000) mx(p-n) (   2695.757  2695.750     0.000) pv lvLSO#pvCtrGdsOflTfbInLso0xc2c err ZMAX_CO 
    NSc::ccr n   3164 p   3157 mn(n-p) (   2697.141  2697.125  3616.440) mx(p-n) (   2697.140  2697.125     0.000) pv lvLSO#pvCtrGdsOflInLso0xbf7425 err 
    NSc::ccr n   3165 p   3157 mn(n-p) (   1766.689  1422.000  3542.000) mx(p-n) (   3299.868  3644.500   349.621) pv     lvLSO#pvOcrGdsPrt0xbf6d0d0 err 
    NSc::ccr n   3166 p   3157 mn(n-p) (   1766.689  1422.000  3727.000) mx(p-n) (   3299.868  3644.500   319.621) pv  lvLSO#pvOcrGdsBotClp0xbfa1610 err 
    NSc::ccr n   3167 p   3157 mn(n-p) (   1666.207  1584.500  3907.798) mx(p-n) (   2442.429  2740.688    18.025) pv lvLSO#pvOcrGdsTfbInLso0xbfa181 err 
    NSc::ccr n   3168 p   3157 mn(n-p) (   1930.553  1585.875  3800.298) mx(p-n) (   3463.729  3808.375    18.025) pv   lvLSO#pvOcrGdsInLso0xbf6d280 err 
    NSc::ccr n   3169 p   3157 mn(n-p) (   2774.027  1062.938     0.000) mx(p-n) (   1643.136  2811.062  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs# err ZMIN_CO 
    NSc::ccr n   3170 p   3157 mn(n-p) (   2833.238  2300.812     0.000) mx(p-n) (    797.491  2737.188  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3171 p   3157 mn(n-p) (   2811.082  2774.000     0.000) mx(p-n) (   1062.991  1643.125  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3172 p   3157 mn(n-p) (   2737.217  2833.250     0.000) mx(p-n) (   2300.790   797.500  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3173 p   3157 mn(n-p) (   1643.137  2811.062     0.000) mx(p-n) (   2774.026  1062.938  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3174 p   3157 mn(n-p) (    797.492  2737.188     0.000) mx(p-n) (   2833.237  2300.812  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3175 p   3157 mn(n-p) (   1062.992  1643.125     0.000) mx(p-n) (   2811.081  2774.000  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3176 p   3157 mn(n-p) (   2300.791   797.500     0.000) mx(p-n) (   2737.216  2833.250  3879.621) pv lvLSO#pvOavBotRibs#OavBotRibs: err ZMIN_CO 
    NSc::ccr n   3177 p   3157 mn(n-p) (   2602.420  2602.438     0.000) mx(p-n) (   2602.419  2602.438  3879.621) pv     lvLSO#pvOavBotHub0xbf21f78 err ZMIN_CO 
    NSc::ccr n   3178 p   3157 mn(n-p) (   2774.025  1322.438   242.000) mx(p-n) (   1810.978  2811.062  3634.621) pv lvLSO#pvIavBotRibs#IavBotRibs# err 
    NSc::ccr n   3179 p   3157 mn(n-p) (   2833.236  2365.562   242.000) mx(p-n) (   1099.626  2737.188  3634.621) pv lvLSO#pvIavBotRibs#IavBotRibs: err 
    NSc::ccr n   3180 p   3157 mn(n-p) (   2811.082  2774.000   242.000) mx(p-n) (   1322.437  1811.000  3634.621) pv lvLSO#pvIavBotRibs#IavBotRibs: err 
    ...
    NSc::ccr n   3192 p   3157 mn(n-p) (   1248.844  2737.188  3542.000) mx(p-n) (   2833.237  2397.562   425.621) pv lvLSO#pvIavTopRibs#IavRibs:5#I err 
    NSc::ccr n   3193 p   3157 mn(n-p) (   1450.566  1893.875  3542.000) mx(p-n) (   2811.081  2774.000   425.621) pv lvLSO#pvIavTopRibs#IavRibs:6#I err 
    NSc::ccr n   3194 p   3157 mn(n-p) (   2397.553  1248.812  3542.000) mx(p-n) (   2737.216  2833.188   425.621) pv lvLSO#pvIavTopRibs#IavRibs:7#I err 
    NSc::ccr n   3195 p   3156 mn(n-p) (   1985.172  1640.500  3993.311) mx(p-n) (   3518.350  3863.000    -5.000) pv lvOAV#pvOcrGdsLsoInOav0xbfa3df err ZMAX_OUT 
    NSc::ccr n   3196 p   3195 mn(n-p) (     24.199    24.188     0.000) mx(p-n) (     24.199    24.188     0.000) pv lvOcrGdsLsoInOav#pvOcrGdsTfbIn err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3197 p   3196 mn(n-p) (      1.383     1.375     0.000) mx(p-n) (      1.383     1.375     0.000) pv lvOcrGdsTfbInOav#pvOcrGdsInOav err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3198 p   3156 mn(n-p) (   3758.264  4210.750  3993.311) mx(p-n) (   1745.258  1292.750    -5.000) pv lvOAV#pvOcrCalLsoInOav0xbfa3eb err ZMAX_OUT 
    NSc::ccr n   3199 p   3155 mn(n-p) (   4784.367  1475.375   625.500) mx(p-n) (   1746.629  5044.688  4125.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    NSc::ccr n   3200 p   3199 mn(n-p) (      4.229     5.375     3.000) mx(p-n) (      5.201     4.250     3.000) pv lvPmtHemi#pvPmtHemiVacuum0xc13 err 
    NSc::ccr n   3201 p   3200 mn(n-p) (    -22.084   106.500   -29.000) mx(p-n) (     84.531   -18.812   -29.000) pv lvPmtHemiVacuum#pvPmtHemiCatho err XMIN_OUT ZMIN_OUT YMAX_OUT ZMAX_OUT 
    NSc::ccr n   3202 p   3200 mn(n-p) (     38.238   102.438     0.000) mx(p-n) (     87.172    44.875     0.000) pv lvPmtHemiVacuum#pvPmtHemiBotto err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3203 p   3200 mn(n-p) (    136.438    46.375    71.500) mx(p-n) (     54.449   142.688    71.500) pv lvPmtHemiVacuum#pvPmtHemiDynod err 
    NSc::ccr n   3204 p   3155 mn(n-p) (   4825.814  1639.250   621.500) mx(p-n) (   1885.295  5094.375  4121.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    NSc::ccr n   3205 p   3155 mn(n-p) (   5188.022  1940.500   625.500) mx(p-n) (   1329.981  4601.938  4125.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    NSc::ccr n   3206 p   3205 mn(n-p) (      4.173     5.062     3.000) mx(p-n) (      5.408     4.188     3.000) pv lvPmtHemi#pvPmtHemiVacuum0xc13 err 
    NSc::ccr n   3207 p   3206 mn(n-p) (    -16.468    69.500   -29.000) mx(p-n) (    118.938   -23.875   -29.000) pv lvPmtHemiVacuum#pvPmtHemiCatho err XMIN_OUT ZMIN_OUT YMAX_OUT ZMAX_OUT 
    NSc::ccr n   3208 p   3206 mn(n-p) (     48.564    76.375     0.000) mx(p-n) (    110.712    33.500     0.000) pv lvPmtHemiVacuum#pvPmtHemiBotto err ZMIN_CO ZMAX_CO 
    NSc::ccr n   3209 p   3206 mn(n-p) (    144.729    58.875    71.500) mx(p-n) (     40.601   130.688    71.500) pv lvPmtHemiVacuum#pvPmtHemiDynod err 
    NSc::ccr n   3210 p   3155 mn(n-p) (   5242.260  2061.375   621.500) mx(p-n) (   1507.689  4637.625  4121.500) pv lvOIL#pvAdPmtArray#pvAdPmtArra err 
    ...
    NSc::ccr n  12225 p   3147 mn(n-p) (  11628.265  1794.938   150.000) mx(p-n) (   2774.523 15480.688   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12226 p   3147 mn(n-p) (  14979.191  4151.750   150.000) mx(p-n) (   1753.470 11326.125   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12227 p   3147 mn(n-p) (  10443.004  8369.250   150.000) mx(p-n) (   1794.919  2774.500   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12228 p   3147 mn(n-p) (   6288.400 16757.875   150.000) mx(p-n) (   7410.776  1753.500   150.000) pv lvNearHallBot#pvNearHallRadSla err 
    NSc::ccr n  12229 p   3147 mn(n-p) (    414.836   414.875  -150.000) mx(p-n) (    414.838   414.875 10150.000) pv lvNearHallBot#pvNearHallRadSla err ZMIN_OUT 
    2017-06-27 20:45:11.361 INFO  [1538289] [NScene::check_containment@506] NScene::check_containment verbosity 1 tot 12230 err 3491 err/tot       0.29



NScene::postimportmesh
-------------------------

Top of the z-bbox is coincident at -5475.5::

    2017-06-27 15:51:06.834 INFO  [1455881] [NScene::postimportmesh@415] NScene::postimportmesh numNd 12230 dbgnode 3159
    2017-06-27 15:51:06.834 INFO  [1455881] [NScene::dumpNd@702] NScene::dumpNd nidx 3159 node exists  verbosity 1

    nd idx/repeatIdx/mesh/nch/depth/nprog  [3159:  0: 36:  0:14:   0] bnd:Acrylic///GdDopedLS
       nd.tr.t  1.000   0.000   0.000   0.000 
                0.000   1.000   0.000   0.000 
                0.000   0.000   1.000   0.000 
                0.000   0.000   7.500   1.000 

      nd.gtr.t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 


     mesh_id 36 meshmeta NScene::meshmeta mesh_id  36 lvidx  22 height  2 soname                        gds0xc28d3f0 lvname      /dd/Geometry/AD/lvGDS0xbf6cbb8
     mesh_idx 36 pmesh_idx 35 root [ 0:un] proot [ 0:un]
        node_t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7100.000   1.000 

       pnode_t  0.543  -0.840   0.000   0.000 
                0.840   0.543   0.000   0.000 
                0.000   0.000   1.000   0.000 
              -18079.453 -799699.438 -7107.500   1.000 

     csg_bb   mi  (-1550.00 -1550.00 -1535.00)  mx  (1550.00 1550.00 1624.44) 
     pcsg_bb  mi  (-1565.00 -1565.00 -1542.50)  mx  (1565.00 1565.00 1631.94) 
     csg_tbb   mi  (-20222.79 -801842.75 -8635.00)  mx  (-15936.12 -797556.12 -5475.56) 
     pcsg_tbb  mi  (-20243.53 -801863.50 -8650.00)  mx  (-15915.38 -797535.38 -5475.56) 
    Assertion failed: (0 && "NScene::postimportmesh HARIKARI"), function postimportmesh, file /Users/blyth/opticks/opticksnpy/NScene.cpp, line 478.
    Process 89361 stopped





Checking the solids individually
-----------------------------------


::

   opticks-tbool 24    # cylinder with conical top hat, with a bit of lip
   opticks-tbool 22    # similar but with hub cap at middle

   opticks-tbool-vi 24
   opticks-tbool-vi 22



        3158 (24)
          |
        3159 (22)  

::

     62 tbool24--(){ cat << EOP
     63 
     64 import logging
     65 log = logging.getLogger(__name__)
     66 from opticks.ana.base import opticks_main
     67 from opticks.analytic.csg import CSG  
     68 args = opticks_main(csgpath="$TMP/tbool/24")
     69 
     70 CSG.boundary = args.testobject
     71 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     72 
     76 a = CSG("cylinder", param = [0.000,0.000,0.000,1560.000],param1 = [-1542.500,1542.500,0.000,0.000])
                                                         r                   z1       z2


     77 b = CSG("cylinder", param = [0.000,0.000,0.000,1565.000],param1 = [3085.000,3100.000,0.000,0.000])   # (5 mm lip in radius, of 15mm height)
                                                         r                   z1      z2
     In [1]: 1542.5*2                                                     1542.500  1557.5   
     Out[1]: 3085.0
           
     78 c = CSG("cone", param = [1520.393,3100.000,100.000,3174.440],param1 = [0.000,0.000,0.000,0.000])
                                     r1    z1       r2      z2        cone starts from 43 mm smaller radius                                 

     79 bc = CSG("union", left=b, right=c)
     80 bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1542.500,1.000]]
     81 
     82 abc = CSG("union", left=a, right=bc)
     86 
     87 
     88 obj = abc

::

     62 tbool22--(){ cat << EOP
     63 
     64 import logging
     65 log = logging.getLogger(__name__)
     66 from opticks.ana.base import opticks_main
     67 from opticks.analytic.csg import CSG  
     68 args = opticks_main(csgpath="$TMP/tbool/22")
     69 
     70 CSG.boundary = args.testobject
     71 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     72 
     75 
     76 a = CSG("cylinder", param = [0.000,0.000,0.000,1550.000],param1 = [-1535.000,1535.000,0.000,0.000])
                                                         r                   z1       z2            
                                             # 10 mm smaller radius       smaller             


     77 b = CSG("cone", param = [1520.000,3070.000,75.000,3145.729],param1 = [0.000,0.000,0.000,0.000])
                                   r1 z1           r2      z2
     78 c = CSG("cylinder", param = [0.000,0.000,0.000,75.000],param1 = [3145.729,3159.440,0.000,0.000])   # hub cap, 
                                                        r                z1       z2
     79 bc = CSG("union", left=b, right=c)
     80 bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1535.000,1.000]]
     81 
     82 abc = CSG("union", left=a, right=bc)
     83 
     87 
     88 obj = abc







OKTest_genstep_targetting
==========================

Fix
----

Fixed by using the gdmlaux_target as the default for genstep and domain targetting 
in addition to the composition targetting that was done already.  

Now get the expected expanding sphere of photons when run::

   OKTest   


Issue
--------

Run OKTest, press A to see photon propagation.  Press P a few times to change photon
rendering use the animation slider to change time.  Use F and slide to change far.

Note that the photons are not coming from center of AD, they appear very very distant
as using Z and moving backwards shows the geometry shrinking away but very few of the
photon lines are moving.  

Also the histories show lots of scatters and the time range of the animation is 
very large.


Lots of scatters and MI shows many photons are ending at edge of world::

    2020-11-26 17:39:11.505 INFO  [126440] [OpticksAttrSeq::dumpTable@422] OpticksIdx::makeHistoryItemIndex seqhis
        0      1377  0.138 666666666d TO SC SC SC SC SC SC SC SC SC 
        1       839  0.084         4d                         TO AB 
        2       746  0.075        46d                      TO SC AB 
        3       726  0.073       366d                   TO SC SC MI 
        4       639  0.064      3666d                TO SC SC SC MI 
        5       600  0.060       466d                   TO SC SC AB 
        6       586  0.059     36666d             TO SC SC SC SC MI 
        7       579  0.058        36d                      TO SC MI 
        8       534  0.053    366666d          TO SC SC SC SC SC MI 
        9       455  0.045      4666d                TO SC SC SC AB 
       10       410  0.041         3d                         TO MI 
       11       403  0.040   3666666d       TO SC SC SC SC SC SC MI 
       12       390  0.039     46666d             TO SC SC SC SC AB 
       13       363  0.036  36666666d    TO SC SC SC SC SC SC SC MI 
       14       333  0.033    466666d          TO SC SC SC SC SC AB 
       15       283  0.028 c66666666d TO SC SC SC SC SC SC SC SC BT 
       16       243  0.024   4666666d       TO SC SC SC SC SC SC AB 
       17       184  0.018  46666666d    TO SC SC SC SC SC SC SC AB 
       18       134  0.013 466666666d TO SC SC SC SC SC SC SC SC AB 
       19        46  0.005       3c6d                   TO SC BT MI 
       20        34  0.003        3cd                      TO BT MI 
       21        23  0.002      3c66d                TO SC SC BT MI 
       22        17  0.002   3c66666d       TO SC SC SC SC SC BT MI 
       23        16  0.002     3c666d             TO SC SC SC BT MI 
       24        14  0.001    3c6666d          TO SC SC SC SC BT MI 
       25        13  0.001  3c666666d    TO SC SC SC SC SC SC BT MI 
       26         8  0.001 cc6666666d TO SC SC SC SC SC SC SC BT BT 
       27         1  0.000     4cc66d             TO SC SC BT BT AB 
       28         1  0.000  366666ccd    TO BT BT SC SC SC SC SC MI 
       29         1  0.000 6666666ccd TO BT BT SC SC SC SC SC SC SC 
       30         1  0.000 666666cc6d TO SC BT BT SC SC SC SC SC SC 
       31         1  0.000 b9b9ccc66d TO SC SC BT BT BT DR BR DR BR 
      TOT     10000


This looks like the genstep targetting is not working ?

Running with --dbgaim shows are defaulting to node 0 

::

   OKTest --gensteptarget 3153 

   ## here the photon propagation looks bizarre shape (not expected expanding sphere) 
   ## because the domaintarget is defaulting to 0 giving very imprecision photon positions

   OKTest --gensteptarget 3153 --domaintarget 3153

   ## get the expected expanding sphere intersecting with detector layers 



::

    epsilon:opticks blyth$ opticks-f gensteptarget 
    ./opticksgeo/OpticksGen.cc:            << " genstepTarget --gensteptarget : " << genstepTarget
    ./ggeo/GNodeLib.cc:    targets.push_back(m_ok->getGenstepTarget());  // --gensteptarget
    ./ggeo/GNodeLib.cc:        << " --gensteptarget " << m_ok->getGenstepTarget() 
    ./optickscore/Opticks.hh:       int         getGenstepTarget() const ;  // --gensteptarget
    ./optickscore/OpticksCfg.cc:    m_gensteptarget(BOpticksResource::DefaultGenstepTarget(0)),   // OPTICKS_GENSTEP_TARGET envvar
    ./optickscore/OpticksCfg.cc:   char gensteptarget[256];
    ./optickscore/OpticksCfg.cc:   snprintf(gensteptarget,256, "Integer controlling where fabricated torch gensteps are located. "
    ./optickscore/OpticksCfg.cc:                               "Default gensteptarget %d can be defined with envvar OPTICKS_GENSTEP_TARGET.", m_gensteptarget );
    ./optickscore/OpticksCfg.cc:       ("gensteptarget",  boost::program_options::value<int>(&m_gensteptarget), gensteptarget );
    ./optickscore/OpticksCfg.cc:    return m_gensteptarget ; 
    ./optickscore/OpticksCfg.hh:     int         m_gensteptarget ;  
    ./optickscore/Opticks.cc:int  Opticks::getGenstepTarget() const  // --gensteptarget, default sensitive to OPTICKS_GENSTEP_TARGET envvar 
    epsilon:opticks blyth$ 






The gdmlaux succeeds to target intended default for the view, but not the genstep ?

::

    123 void OpticksAim::setupCompositionTargetting()
    124 {
    125     bool autocam = true ;
    126     unsigned deferred_target = getTargetDeferred();   // default to 0   
    127     // ^^^^^^^^^^^^^ suspect no longer needed
    128 
    129     unsigned cmdline_target = m_ok->getTarget();      // sensitive to OPTICKS_TARGET envvar, fallback 0 
    130 
    131     const char* target_lvname = m_ok->getGDMLAuxTargetLVName() ;
    132     int gdmlaux_target =  m_ggeo ? m_ggeo->getFirstNodeIndexForGDMLAuxTargetLVName() : -1 ;  // sensitive to GDML auxilary lvname metadata (label, target)  
    133 
    134     unsigned active_target = 0 ;
    135 
    136     if( cmdline_target > 0 )
    137     {
    138         active_target = cmdline_target ;
    139     }
    140     else if( deferred_target > 0 )
    141     {
    142         active_target = deferred_target ;
    143     }
    144     else if( gdmlaux_target > 0 )
    145     {
    146         active_target = gdmlaux_target ;
    147     }



::

    417 TorchStepNPY* OpticksGen::makeTorchstep(unsigned gencode)
    418 {
    419     assert( gencode == OpticksGenstep_TORCH );
    420 
    421     TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep(gencode);
    422 
    423     if(torchstep->isDefault())
    424     {   
    425         int frameIdx = torchstep->getFrameIndex(); 
    426         int detectorDefaultFrame = m_ok->getDefaultFrame() ; 
    427         int gdmlaux_target =  m_ggeo ? m_ggeo->getFirstNodeIndexForGDMLAuxTargetLVName() : -1 ;  // sensitive to GDML auxilary lvname metadata (label, target) 
    428         int cmdline_target = m_ok->getGenstepTarget() ;   // --gensteptarget
    429         
    430         unsigned active_target = 0 ;
    431             
    432         if( cmdline_target > 0 )
    433         {   
    434             active_target = cmdline_target ;
    435         }
    436         else if( gdmlaux_target > 0 )
    437         {   
    438             active_target = gdmlaux_target ;
    439         }
    440         
    441         LOG(error) 
    442             << " as torchstep isDefault replacing placeholder frame "
    443             << " frameIdx : " << frameIdx 
    444             << " detectorDefaultFrame : " << detectorDefaultFrame
    445             << " cmdline_target [--gensteptarget] : " << cmdline_target
    446             << " gdmlaux_target : " << gdmlaux_target
    447             << " active_target : " << active_target
    448             ;
    449         
    450         torchstep->setFrame(active_target);
    451     }
    452 



    063 void OpticksAim::registerGeometry(GGeo* ggeo)
     64 {   
     65     assert( ggeo );
     66     m_ggeo = ggeo ;
     67     
     68     const char* gdmlaux_target_lvname = m_ok->getGDMLAuxTargetLVName() ;
     69     m_gdmlaux_target =  m_ggeo->getFirstNodeIndexForGDMLAuxTargetLVName() ; // sensitive to GDML auxilary lvname metadata (label, target)  
     70     
     71     int cmdline_domaintarget = m_ok->getDomainTarget();    // --domaintarget 
     72     
     73     unsigned active_domaintarget = 0 ;
     74     if( cmdline_domaintarget > 0 )
     75     {   
     76         active_domaintarget = cmdline_domaintarget ;
     77     } 
     78     else if( m_gdmlaux_target > 0 )
     79     {   
     80         active_domaintarget = m_gdmlaux_target ;
     81     }
     82     
     83     glm::vec4 center_extent = m_ggeo->getCE(active_domaintarget);
     84     
     85     LOG(LEVEL)
     86         << " setting SpaceDomain : " 
     87         << " cmdline_domaintarget [--domaintarget] " << cmdline_domaintarget
     88         << " gdmlaux_target " << m_gdmlaux_target
     89         << " gdmlaux_target_lvname  " << gdmlaux_target_lvname
     90         << " active_domaintarget " << active_domaintarget
     91         << " center_extent " << gformat(center_extent)
     92         ;
     93     
     94     m_ok->setSpaceDomain( center_extent );
     95 }




::

    OKTest --dbgaim
    ...

    2020-11-26 20:27:46.003 INFO  [640181] [Scene::uploadGeometry@803]  nmm 6
    2020-11-26 20:27:46.003 INFO  [640181] [RContext::initUniformBuffer@59] RContext::initUniformBuffer
    2020-11-26 20:27:46.071 ERROR [640181] [OpticksAim::setupCompositionTargetting@149]  cmdline_target 0 gdmlaux_target 3153 active_target 3153
    2020-11-26 20:27:46.071 INFO  [640181] [GNodeLib::dumpVolumes@753] OpticksAim::setTarget num_volumes 12230 --target 0 --domaintarget 0 --gensteptarget 0 cursor 3153
                active_composition :       3153
                     active_domain :       3153
               cmdline_composition :          0
                    cmdline_domain :          0
               gdmlaux_composition :       3153
                    gdmlaux_domain :       3153
    2020-11-26 20:27:46.071 INFO  [640181] [GNodeLib::dumpVolumes@783] first volumes 
              0                                               World0xc15cfc00x40f7000        ce   0.000   0.000   0.000 2400000.000 
              1                   /dd/Geometry/Sites/lvNearSiteRock0xc0303500x40f6d90        ce -16520.000 -802110.000 3892.925 34569.875 
              2                    /dd/Geometry/Sites/lvNearHallTop0xc1368900x3ee49d0        ce -12841.452 -806876.000 5390.000 22545.344 
              3             /dd/Geometry/PoolDetails/lvNearTopCover0xc1370600x3ebf2d0        ce -16520.098 -802110.000 -2088.000 7801.031 
              4                           /dd/Geometry/RPC/lvRPCMod0xbf54e600x3ecba70        ce -11612.390 -799007.250 683.903 1509.703 
              5                          /dd/Geometry/RPC/lvRPCFoam0xc032c880x3ecb480        ce -11611.268 -799018.375 683.903 1455.636 
              6                     /dd/Geometry/RPC/lvRPCBarCham140xbf4c6a00x3eca7f0        ce -11611.268 -799018.375 669.903 1448.750 
              7                      /dd/Geometry/RPC/lvRPCGasgap140xbf98ae00x3ec5870        ce -11611.268 -799018.375 669.903 1434.939 
              8                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11124.673 -799787.375 669.903 948.345 
              9                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11263.700 -799567.625 669.903 948.345 
             10                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11402.727 -799347.938 669.903 948.345 
             11                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11541.754 -799128.250 669.903 948.345 
             12                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11680.781 -798908.500 669.903 948.345 
             13                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11819.809 -798688.812 669.903 948.345 
             14                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11958.835 -798469.125 669.903 948.345 
             15                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -12097.862 -798249.375 669.903 948.345 
             16                     /dd/Geometry/RPC/lvRPCBarCham140xbf4c6a00x3eca7f0        ce -11611.268 -799018.375 707.903 1448.750 
             17                      /dd/Geometry/RPC/lvRPCGasgap140xbf98ae00x3ec5870        ce -11611.268 -799018.375 707.903 1434.939 
             18                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11124.673 -799787.375 707.903 948.345 
             19                         /dd/Geometry/RPC/lvRPCStrip0xc2213c00x3ec5750        ce -11263.700 -799567.625 707.903 948.345 
    2020-11-26 20:27:46.072 INFO  [640181] [GNodeLib::dumpVolumes@792] targetted volumes(**) OR volumes with extent greater than 5000 mm 
     **       0                                               World0xc15cfc00x40f7000        ce   0.000   0.000   0.000 2400000.000 
              1                   /dd/Geometry/Sites/lvNearSiteRock0xc0303500x40f6d90        ce -16520.000 -802110.000 3892.925 34569.875 
              2                    /dd/Geometry/Sites/lvNearHallTop0xc1368900x3ee49d0        ce -12841.452 -806876.000 5390.000 22545.344 
              3             /dd/Geometry/PoolDetails/lvNearTopCover0xc1370600x3ebf2d0        ce -16520.098 -802110.000 -2088.000 7801.031 
             88                      /dd/Geometry/RPC/lvNearRPCRoof0xbf400300x3ecbc40        ce -16544.561 -802110.000 -1288.616 10993.875 
           2357            /dd/Geometry/RPCSupport/lvNearRPCSptRoof0xc2c55e00x3ee42f0        ce -16544.561 -802110.000 -1583.371 10801.312 
           2358        /dd/Geometry/RPCSupport/lvNearHbeamSmallUnit0xc5bef700x3ed1810        ce -20858.879 -795373.062 -1583.371 5570.959 
           2431          /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a9880x3ed71c0        ce -19241.010 -797899.375 -1583.371 6110.248 
           2610          /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a9880x3ed71c0        ce -17083.850 -801267.875 -1583.371 6110.248 
           2789          /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a9880x3ed71c0        ce -14926.691 -804636.375 -1583.371 6110.248 
           2968          /dd/Geometry/RPCSupport/lvNearHbeamBigUnit0xbf3a9880x3ed71c0        ce -12769.532 -808004.812 -1583.371 6110.249 
           3147                    /dd/Geometry/Sites/lvNearHallBot0xbf89c600x412b0d0        ce -16520.000 -802110.000 -7260.000 9847.688 
           3148                    /dd/Geometry/Pool/lvNearPoolDead0xc2dc4900x4129a30        ce -16520.164 -802109.938 -7110.000 7801.125 
           3149                   /dd/Geometry/Pool/lvNearPoolLiner0xc21e9d00x4128d30        ce -16520.162 -802109.938 -7068.000 7711.688 
           3150                     /dd/Geometry/Pool/lvNearPoolOWS0xbf938400x3fa90b0        ce -16520.000 -802110.000 -7066.000 9316.688 
           3151                 /dd/Geometry/Pool/lvNearPoolCurtain0xc2ceef00x3fa6cc0        ce -16520.145 -802110.000 -6566.000 6642.781 
           3152                     /dd/Geometry/Pool/lvNearPoolIWS0xc28bc600x3efba00        ce -16520.000 -802110.000 -6564.000 7928.375 
     **    3153                               /dd/Geometry/AD/lvADE0xc2a78c00x3ef9140        ce -18079.453 -799699.438 -6605.000 3005.000 
          12229                /dd/Geometry/RadSlabs/lvNearRadSlab90xc15c2080x412afb0        ce -16520.098 -802110.000 -12410.000 7801.031 
    2020-11-26 20:27:46.074 NONE  [640181] [OpticksViz::uploadGeometry@384] ]




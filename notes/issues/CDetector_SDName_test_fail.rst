CDetector_SDName_test_fail
============================


Axel reports::

    Hi Simon,

    I found some time now to have a closer look to the errors:

     3  /30  Test #3  : CFG4Test.CTestDetectorTest                    ***Exception: Child aborted    1.28   
      7  /30  Test #7  : CFG4Test.CG4Test                              ***Exception: Child aborted    1.22   
      21 /30  Test #21 : CFG4Test.CInterpolationTest                   ***Exception: Child aborted    1.08   
      26 /30  Test #26 : CFG4Test.CRandomEngineTest                    ***Exception: Child aborted    1.14   
      1  /1   Test #1  : OKG4Test.OKG4Test                             ***Exception: Child aborted    1.41   

    They all have the same error when running separately:

    OKG4Test: /home/gpu/opticks/cfg4/CDetector.cc:141: void CDetector::hookupSD(): Assertion `strcmp( sdname.c_str(), sdn ) == 0' failed.
    I looked at the code at /home/gpu/opticks/cfg4/CDetector.cc:141:
    assert( strcmp( sdname.c_str(), sdn ) == 0 ) ;  

    I let the two strings write to the console and got:

    2018-08-28 15:28:00.001 ERROR [6809] [CDetector::hookupSD@141]  sdn SD_AssimpGGeo sdname SD0

    At this level I got stuck as I don't really understand yet the entire code.

    Do you also get this error?

    Best regards,

    Axel


::

    123 void CDetector::hookupSD()
    124 {
    125     unsigned nlvsd = m_ggeo->getNumLVSD() ;
    126     const std::string sdname = m_sd ? m_sd->GetName() : "noSD" ;
    127     LOG(error)
    128         << " nlvsd " << nlvsd
    129         << " sd " << m_sd
    130         << " sdname " << sdname
    131         ;
    132 
    133 
    134     if(!m_sd) return ;
    135     for( unsigned i = 0 ; i < nlvsd ; i++)
    136     {
    137         std::pair<std::string,std::string> lvsd = m_ggeo->getLVSD(i) ;
    138         const char* lvn = lvsd.first.c_str();
    139         const char* sdn = lvsd.second.c_str();
    140 
    141         assert( strcmp( sdname.c_str(), sdn ) == 0 ) ;
    142 
    143         //const char* lvn = m_ggeo->getCathodeLV(i); 
    144 
    145         const G4LogicalVolume* lv = m_traverser->getLV(lvn);
    146 
    147         LOG(error)
    148              << "SetSensitiveDetector"
    149              << " lvn " << lvn
    150              << " sdn " << sdn
    151              << " lv " << lv
    152              ;
    153 
    154         if(!lv) LOG(fatal) << " no lv " << lvn ;
    155         assert(lv);
    156 
    157         const_cast<G4LogicalVolume*>(lv)->SetSensitiveDetector(m_sd) ;
    158     }
    159 }



This is curious, as I dont see the fail. However, you are doing better than me : I get nlvsd 0

    2018-08-28 22:26:08.105 ERROR [2989735] [CDetector::hookupSD@127]  nlvsd 0 sd 0x10f12c210 sdname SD0

So I tried updating my geocache `op.sh -G` to see if that gets us to the same error, and I do::

    lldb CG4Test 
    ...
    2018-08-28 22:28:46.349 INFO  [2996153] [CSurfaceLib::convert@136] CSurfaceLib  numBorderSurface 8 numSkinSurface 34
    2018-08-28 22:28:46.349 INFO  [2996153] [CDetector::attachSurfaces@339] [--dbgsurf] CDetector::attachSurfaces DONE 
    2018-08-28 22:28:46.349 ERROR [2996153] [CDetector::hookupSD@127]  nlvsd 2 sd 0x10ea45320 sdname SD0
    Assertion failed: (strcmp( sdname.c_str(), sdn ) == 0), function hookupSD, file /Users/blyth/opticks/cfg4/CDetector.cc, line 141.
    ...
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001ccd28 libCFG4.dylib`CDetector::hookupSD(this=0x000000010ea46160) at CDetector.cc:141
        frame #5: 0x00000001001d3815 libCFG4.dylib`CGDMLDetector::init(this=0x000000010ea46160) at CGDMLDetector.cc:78
        frame #6: 0x00000001001d34d3 libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010ea46160, hub=0x00007ffeefbfe658, query=0x000000010a725560, sd=0x000000010ea45320) at CGDMLDetector.cc:40
        frame #7: 0x00000001001d385d libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010ea46160, hub=0x00007ffeefbfe658, query=0x000000010a725560, sd=0x000000010ea45320) at CGDMLDetector.cc:38
        frame #8: 0x0000000100127b9d libCFG4.dylib`CGeometry::init(this=0x000000010ea46110) at CGeometry.cc:68
        frame #9: 0x000000010012789c libCFG4.dylib`CGeometry::CGeometry(this=0x000000010ea46110, hub=0x00007ffeefbfe658, sd=0x000000010ea45320) at CGeometry.cc:51
        frame #10: 0x0000000100127c35 libCFG4.dylib`CGeometry::CGeometry(this=0x000000010ea46110, hub=0x00007ffeefbfe658, sd=0x000000010ea45320) at CGeometry.cc:50
        frame #11: 0x00000001001f160c libCFG4.dylib`CG4::CG4(this=0x000000010e5a1160, hub=0x00007ffeefbfe658) at CG4.cc:120
        frame #12: 0x00000001001f1f8d libCFG4.dylib`CG4::CG4(this=0x000000010e5a1160, hub=0x00007ffeefbfe658) at CG4.cc:140
        frame #13: 0x000000010000f113 CG4Test`main(argc=1, argv=0x00007ffeefbfea30) at CG4Test.cc:31
        frame #14: 0x00007fff7acac015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x00000001001ccd28 libCFG4.dylib`CDetector::hookupSD(this=0x000000010ea46160) at CDetector.cc:141
       138 	        const char* lvn = lvsd.first.c_str(); 
       139 	        const char* sdn = lvsd.second.c_str(); 
       140 	
    -> 141 	        assert( strcmp( sdname.c_str(), sdn ) == 0 ) ;  
       142 	
       143 	        //const char* lvn = m_ggeo->getCathodeLV(i); 
       144 	
    (lldb) p lvsd
    (std::__1::pair<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >) $0 = (first = "__dd__Geometry__PMT__lvHeadonPmtCathode0xc2c8d98", second = "SD_AssimpGGeo")
    (lldb) p 
         



Removing the assert moves along to another error::

    2018-08-28 22:39:00.301 INFO  [3004244] [CDetector::attachSurfaces@339] [--dbgsurf] CDetector::attachSurfaces DONE 
    2018-08-28 22:39:00.301 ERROR [3004244] [CDetector::hookupSD@127]  nlvsd 2 sd 0x7fe113e01df0 sdname SD0
    2018-08-28 22:39:00.301 ERROR [3004244] [CDetector::hookupSD@147] SetSensitiveDetector lvn __dd__Geometry__PMT__lvHeadonPmtCathode0xc2c8d98 sdn SD_AssimpGGeo lv 0x0
    2018-08-28 22:39:00.301 FATAL [3004244] [CDetector::hookupSD@154]  no lv __dd__Geometry__PMT__lvHeadonPmtCathode0xc2c8d98
    Assertion failed: (lv), function hookupSD, file /Users/blyth/opticks/cfg4/CDetector.cc, line 155.
    Abort trap: 6
    epsilon:cfg4 blyth$ 


The route cause of the problem, is that I have recently been working on a simple test geometry over in examples/Geant4/CerenkovMinimal
and it seems the changes I made to CDetector are not working with the full geometry.  




Review the stack::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #3: 0x00007fff7ad201ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001001d10c9 libCFG4.dylib`CDetector::hookupSD(this=0x000000010a7e3480) at CDetector.cc:155
        frame #5: 0x00000001001d7835 libCFG4.dylib`CGDMLDetector::init(this=0x000000010a7e3480) at CGDMLDetector.cc:78
        frame #6: 0x00000001001d74f3 libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a7e3480, hub=0x00007ffeefbfe250, query=0x000000010c010bd0, sd=0x000000010a7e2640) at CGDMLDetector.cc:40
        frame #7: 0x00000001001d787d libCFG4.dylib`CGDMLDetector::CGDMLDetector(this=0x000000010a7e3480, hub=0x00007ffeefbfe250, query=0x000000010c010bd0, sd=0x000000010a7e2640) at CGDMLDetector.cc:38
        frame #8: 0x000000010012bd2d libCFG4.dylib`CGeometry::init(this=0x000000010a7e3430) at CGeometry.cc:68
        frame #9: 0x000000010012ba2c libCFG4.dylib`CGeometry::CGeometry(this=0x000000010a7e3430, hub=0x00007ffeefbfe250, sd=0x000000010a7e2640) at CGeometry.cc:51
        frame #10: 0x000000010012bdc5 libCFG4.dylib`CGeometry::CGeometry(this=0x000000010a7e3430, hub=0x00007ffeefbfe250, sd=0x000000010a7e2640) at CGeometry.cc:50
        frame #11: 0x00000001001f562c libCFG4.dylib`CG4::CG4(this=0x00007ffeefbfe090, hub=0x00007ffeefbfe250) at CG4.cc:120
        frame #12: 0x00000001001f5fad libCFG4.dylib`CG4::CG4(this=0x00007ffeefbfe090, hub=0x00007ffeefbfe250) at CG4.cc:140
        frame #13: 0x000000010000ee91 CInterpolationTest`main(argc=1, argv=0x00007ffeefbfea18) at CInterpolationTest.cc:57


    (lldb) f 11
    frame #11: 0x00000001001f562c libCFG4.dylib`CG4::CG4(this=0x00007ffeefbfe090, hub=0x00007ffeefbfe250) at CG4.cc:120
       117 	    m_physics(new CPhysics(this)),
       118 	    m_runManager(m_physics->getRunManager()),
       119 	    m_sd(new CSensitiveDetector("SD0")),
    -> 120 	    m_geometry(new CGeometry(m_hub, m_sd)),
       121 	    m_hookup(m_geometry->hookup(this)),
       122 	    m_mlib(m_geometry->getMaterialLib()),
       123 	    m_detector(m_geometry->getDetector()),

    (lldb) f 8
    frame #8: 0x000000010012bd2d libCFG4.dylib`CGeometry::init(this=0x000000010a7e3430) at CGeometry.cc:68
       65  	        // no options here: will load the .gdml sidecar of the geocache .dae 
       66  	        LOG(fatal) << "CGeometry::init G4 GDML geometry " ; 
       67  	        OpticksQuery* query = m_ok->getQuery();
    -> 68  	        detector  = static_cast<CDetector*>(new CGDMLDetector(m_hub, query, m_sd)) ; 
       69  	    }
       70  	
       71  	    // detector->attachSurfaces();  moved into the ::init of CTestDetector and CGDMLDetector to avoid omission



Traverser has names like below::

    (lldb) p m_traverser->m_lvm
    (std::__1::map<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, const G4LogicalVolume *, std::__1::less<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::allocator<std::__1::pair<const std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, const G4LogicalVolume *> > >) $2 = size=249 {
      [0] = {
        first = "/dd/Geometry/AD/lvADE0xc2a78c0"
        second = 0x000000010efc7760
      }
      [1] = {
        first = "/dd/Geometry/AD/lvGDS0xbf6cbb8"
        second = 0x000000010edfbed0
      }
      [2] = {
        first = "/dd/Geometry/AD/lvIAV0xc404ee8"
        second = 0x000000010edfbfd0
      }


Are missing some name translation::

    (lldb) p lvn
    (const char *) $3 = 0x000000010f1f2610 "__dd__Geometry__PMT__lvHeadonPmtCathode0xc2c8d98"
    (lldb) p m_traverser->getLV("/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98")
    (G4LogicalVolume *) $4 = 0x00000001118046a0
    (lldb) 

Huh the name with "__dd__" was for G4DAE XML.  


lvn is coming from LVSD::

     306 void GGeo::addLVSD(const char* lv, const char* sd)
     307 {
     308    assert( lv ) ;
     309    m_cathode_lv.insert(lv);
     310 
     311    if(sd)
     312    {
     313        if(m_lv2sd == NULL ) m_lv2sd = new NMeta ;
     314        m_lv2sd->set<std::string>(lv, sd) ;
     315    }
     316 }
     317 unsigned GGeo::getNumLVSD() const
     318 {
     319    return m_lv2sd ? m_lv2sd->getNumKeys() : 0 ;
     320 }
     321 std::pair<std::string,std::string> GGeo::getLVSD(unsigned idx) const
     322 {
     323     const char* lv = m_lv2sd->getKey(idx) ;
     324     std::string sd = m_lv2sd->get<std::string>(lv);
     325     return std::pair<std::string,std::string>( lv, sd );
     326 }


The lvsd names are direct from assimp (from the G4DAE) without any translation::

     673 void AssimpGGeo::convertSensorsVisit(GGeo* gg, AssimpNode* node, unsigned int depth)
     674 {
     675     unsigned int nodeIndex = node->getIndex();
     676     const char* lv   = node->getName(0);
     677     //const char* pv   = node->getName(1); 
     678     unsigned int mti = node->getMaterialIndex() ;
     679     GMaterial* mt = gg->getMaterial(mti);
     680     assert( mt );
     681 
     682     /*
     683     NSensor* sensor0 = sens->getSensor( nodeIndex ); 
     684     NSensor* sensor1 = sens->findSensorForNode( nodeIndex ); 
     685     assert(sensor0 == sensor1);
     686     // these do not match
     687     */
     688 
     689     NSensor* sensor = m_sensor_list ? m_sensor_list->findSensorForNode( nodeIndex ) : NULL ;
     690 
     691     GMaterial* cathode = gg->getCathode() ;
     692 
     693     const char* cathode_material_name = gg->getCathodeMaterialName() ;
     694     const char* name = mt->getName() ;
     695     bool name_match = strcmp(name, cathode_material_name) == 0 ;
     696     bool ptr_match = mt == cathode ;   // <--- always false 
     697 
     698     const char* sd = "SD_AssimpGGeo" ;
     699 
     700     if(sensor && name_match)
     701     {
     702          LOG(debug) << "AssimpGGeo::convertSensorsVisit "
     703                    << " depth " << depth
     704                    << " lv " << lv
     705                    << " sd " << sd
     706                    << " ptr_match " << ptr_match
     707                    ;
     708          gg->addLVSD(lv, sd) ;
     709     }
     710 }



Are missing::

     94 char* BStr::DAEIdToG4( const char* daeid, bool trimPtr)
     95 {
     96     /**
     97         Convert daeid such as  "__dd__Geometry__PoolDetails__lvLegInIWSTub0xc400e40" 
     98         to G4 name                  /dd/Geometry/PoolDetails/lvLegInIWSTub
     99     **/







And the CTraverser names are direct from GDML::

    193 void CTraverser::AncestorVisit(std::vector<const G4VPhysicalVolume*> ancestors, bool selected)
    194 {   
    195     G4Transform3D T ;
    196     
    197     for(unsigned int i=0 ; i < ancestors.size() ; i++)
    198     {   
    199         const G4VPhysicalVolume* apv = ancestors[i] ;
    200         
    201         G4RotationMatrix rot, invrot;
    202         if (apv->GetFrameRotation() != 0)
    203         {   
    204             rot = *(apv->GetFrameRotation());
    205             invrot = rot.inverse();
    206         }
    207         
    208         G4Transform3D P(invrot,apv->GetObjectTranslation());
    209         
    210         T = T*P ;
    211     }
    212     const G4VPhysicalVolume* pv = ancestors.back() ; 
    213     const G4LogicalVolume* lv = pv->GetLogicalVolume() ;
    214 
    215     
    216     const std::string& pvn = pv->GetName();
    217     const std::string& lvn = lv->GetName();
    218     
    219     LOG(verbose) << " pvn " << pvn ;
    220     LOG(verbose) << " lvn " << lvn ;
    221 
    222     
    223     updateBoundingBox(lv->GetSolid(), T, selected);
    224     
    225     LOG(debug) << "CTraverser::AncestorVisit " 
    226               << " size " << std::setw(3) << ancestors.size()
    227               << " gcount " << std::setw(6) << m_gcount
    228               << " pvname " << pv->GetName()
    229               ; 
    230     m_gcount += 1 ;
    231 
    232     
    233     collectTransformT(m_gtransforms, T );
    234     m_pvnames.push_back(pvn);
    235     
    236     m_pvs.push_back(pv);  
    237     m_lvs.push_back(lv);  // <-- hmm will be many of the same lv in m_lvs 
    238     
    239     m_lvm[lvn] = lv ;
    240 
    241 
    242     m_ancestor_index += 1 ;
    243 }




Note other DAE names::

    2018-08-29 20:02:16.334 INFO  [3660017] [GSurfaceLib::dumpSkinSurface@1286] GGeo::dumpSkinSurface
     SS    0 :                     NearPoolCoverSurface : __dd__Geometry__PoolDetails__lvNearTopCover0xc137060
     SS    1 :                             RSOilSurface : __dd__Geometry__AdDetails__lvRadialShieldUnit0xc3d7ec0
     SS    2 :                       AdCableTraySurface : __dd__Geometry__AdDetails__lvAdVertiCableTray0xc3a27f0
     SS    3 :                      PmtMtTopRingSurface : __dd__Geometry__PMT__lvPmtTopRing0xc3486f0
     SS    4 :                     PmtMtBaseRingSurface : __dd__Geometry__PMT__lvPmtBaseRing0xc00f400



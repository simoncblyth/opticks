g4-1062-geocache-create-reflectivity-assert
=============================================

::

    epsilon:opticks charles$ geocache-create -D
    === o-cmdline-parse 1 : START
    === o-cmdline-specials 1 :
    === o-cmdline-specials 1 :
    === o-cmdline-binary-match 1 : finding 1st argument with associated binary
    === o-cmdline-binary-match 1 : --okx4test
    === o-cmdline-parse 1 : DONE

    2020-12-20 19:13:31.787 INFO  [6787389] [X4PhysicalVolume::convertMaterials@255]  num_materials 36 num_material_with_efficiency 1
    2020-12-20 19:13:31.787 INFO  [6787389] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 595.
    Process 73978 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.

    Process 73978 launched: '/Users/charles/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff77f3f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff77cd01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff77c981ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010cc0be84 libGGeo.dylib`GSurfaceLib::createStandardSurface(this=0x0000000111b17690, src=0x0000000111a68690) at GSurfaceLib.cc:595
        frame #5: 0x000000010cc0ae42 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111b17690, surf=0x0000000111a68690) at GSurfaceLib.cc:486
        frame #6: 0x000000010cc0ad84 libGGeo.dylib`GSurfaceLib::addBorderSurface(this=0x0000000111b17690, surf=0x0000000111a68690, pv1="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0", pv2="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720", direct=false) at GSurfaceLib.cc:374
        frame #7: 0x000000010cc0aa48 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111b17690, raw=0x0000000111a68690) at GSurfaceLib.cc:358
        frame #8: 0x00000001038ba51e libExtG4.dylib`X4LogicalBorderSurfaceTable::init(this=0x00007ffeefbfd478) at X4LogicalBorderSurfaceTable.cc:66
        frame #9: 0x00000001038ba1d4 libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd478, dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:45
        frame #10: 0x00000001038ba18d libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd478, dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:44
        frame #11: 0x00000001038ba15c libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:37
        frame #12: 0x00000001038c6f63 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfe558) at X4PhysicalVolume.cc:282
        frame #13: 0x00000001038c670f libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe558) at X4PhysicalVolume.cc:192
        frame #14: 0x00000001038c63f5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe558, ggeo=0x0000000111b14760, top=0x0000000118d44660) at X4PhysicalVolume.cc:177
        frame #15: 0x00000001038c56b5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe558, ggeo=0x0000000111b14760, top=0x0000000118d44660) at X4PhysicalVolume.cc:168
        frame #16: 0x0000000100015707 OKX4Test`main(argc=15, argv=0x00007ffeefbfed58) at OKX4Test.cc:108
        frame #17: 0x00007fff77c24015 libdyld.dylib`start + 1
        frame #18: 0x00007fff77c24015 libdyld.dylib`start + 1
    (lldb) 



::

     185 void X4PhysicalVolume::init()
     186 {
     187     LOG(LEVEL) << "[" ;
     188     LOG(LEVEL) << " query : " << m_query->desc() ;
     189 
     190 
     191     convertMaterials();   // populate GMaterialLib
     192     convertSurfaces();    // populate GSurfaceLib
     193     closeSurfaces();
     194     convertSolids();      // populate GMeshLib with GMesh converted from each G4VSolid (postorder traverse processing first occurrence of G4LogicalVolume)  
     195     convertStructure();   // populate GNodeLib with GVolume converted from each G4VPhysicalVolume (preorder traverse) 
     196     convertCheck();
     197 
     198     postConvert();
     199 
     200     LOG(LEVEL) << "]" ;
     201 }

     275 void X4PhysicalVolume::convertSurfaces()
     276 {
     277     LOG(LEVEL) << "[" ;
     278 
     279     size_t num_surf0 = m_slib->getNumSurfaces() ;
     280     assert( num_surf0 == 0 );
     281 
     282     X4LogicalBorderSurfaceTable::Convert(m_slib);
     283     size_t num_lbs = m_slib->getNumSurfaces() ;
     284 
     285     X4LogicalSkinSurfaceTable::Convert(m_slib);
     286     size_t num_sks = m_slib->getNumSurfaces() - num_lbs ;
     287 
     288     m_slib->addPerfectSurfaces();
     289     m_slib->dumpSurfaces("X4PhysicalVolume::convertSurfaces");
     290 
     291     m_slib->collectSensorIndices();
     292     m_slib->dumpSensorIndices("X4PhysicalVolume::convertSurfaces");
     293 
     294     LOG(LEVEL) 
     295         << "]" 
     296         << " num_lbs " << num_lbs
     297         << " num_sks " << num_sks
     298         ;  
     299 
     300 }

     40 X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(GSurfaceLib* dst )
     41     :
     42     m_src(G4LogicalBorderSurface::GetSurfaceTable()),
     43     m_dst(dst)
     44 {
     45     init();
     46 }
     47 
     48 
     49 void X4LogicalBorderSurfaceTable::init()
     50 {
     51     unsigned num_src = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
     52     assert( num_src == m_src->size() );
     53 
     54     LOG(LEVEL) << " NumberOfBorderSurfaces " << num_src ;
     55 
     56     for(size_t i=0 ; i < m_src->size() ; i++)
     57     {
     58         G4LogicalBorderSurface* src = (*m_src)[i] ;
     59 
     60         LOG(LEVEL) << src->GetName() ;
     61 
     62         GBorderSurface* dst = X4LogicalBorderSurface::Convert( src );
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ convert missing REFLECTIVITY with 1062 ??  
     63 
     64         assert( dst );
     65 
     66         m_dst->add(dst) ; // GSurfaceLib
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     67     }
     68 }


Darwin.charles.1062::

    X4PhysicalVolume=INFO X4LogicalBorderSurfaceTable=INFO geocache-create -D 
    ...
    2020-12-20 19:30:17.643 INFO  [6804160] [X4PhysicalVolume::init@187] [
    2020-12-20 19:30:17.643 INFO  [6804160] [X4PhysicalVolume::init@188]  query :  queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2020-12-20 19:30:17.648 INFO  [6804160] [X4PhysicalVolume::convertMaterials@255]  num_materials 36 num_material_with_efficiency 1
    2020-12-20 19:30:17.648 INFO  [6804160] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    2020-12-20 19:30:17.648 INFO  [6804160] [X4PhysicalVolume::convertSurfaces@277] [
    2020-12-20 19:30:17.648 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@54]  NumberOfBorderSurfaces 10
    2020-12-20 19:30:17.648 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceTop
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceBot
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SSTOilSurface
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear1
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear2
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] NearIWSCurtainSurface
    2020-12-20 19:30:17.650 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] NearOWSLinerSurface
    2020-12-20 19:30:17.650 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] NearDeadLinerSurface
    2020-12-20 19:30:17.650 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 595.
    ...
    (lldb) bt
        frame #3: 0x00007fff77c981ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010cc0be84 libGGeo.dylib`GSurfaceLib::createStandardSurface(this=0x0000000111ac5540, src=0x0000000111bb9160) at GSurfaceLib.cc:595
        frame #5: 0x000000010cc0ae42 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111ac5540, surf=0x0000000111bb9160) at GSurfaceLib.cc:486
        frame #6: 0x000000010cc0ad84 libGGeo.dylib`GSurfaceLib::addBorderSurface(this=0x0000000111ac5540, surf=0x0000000111bb9160, pv1="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0", pv2="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720", direct=false) at GSurfaceLib.cc:374
        frame #7: 0x000000010cc0aa48 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111ac5540, raw=0x0000000111bb9160) at GSurfaceLib.cc:358
        frame #8: 0x00000001038ba5ee libExtG4.dylib`X4LogicalBorderSurfaceTable::init(this=0x00007ffeefbfd5b8) at X4LogicalBorderSurfaceTable.cc:66
        frame #9: 0x00000001038ba2a4 libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd5b8, dst=0x0000000111ac5540) at X4LogicalBorderSurfaceTable.cc:45
        frame #10: 0x00000001038ba25d libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd5b8, dst=0x0000000111ac5540) at X4LogicalBorderSurfaceTable.cc:44
        frame #11: 0x00000001038ba22c libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(dst=0x0000000111ac5540) at X4LogicalBorderSurfaceTable.cc:37
        frame #12: 0x00000001038c7030 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfe518) at X4PhysicalVolume.cc:282
        frame #13: 0x00000001038c67df libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe518) at X4PhysicalVolume.cc:192
        frame #14: 0x00000001038c64c5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe518, ggeo=0x0000000111ac25f0, top=0x0000000111b8cee0) at X4PhysicalVolume.cc:177
        frame #15: 0x00000001038c5785 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe518, ggeo=0x0000000111ac25f0, top=0x0000000111b8cee0) at X4PhysicalVolume.cc:168
        frame #16: 0x0000000100015707 OKX4Test`main(argc=15, argv=0x00007ffeefbfed10) at OKX4Test.cc:108
        frame #17: 0x00007fff77c24015 libdyld.dylib`start + 1
    (lldb) 

::

     483 void GSurfaceLib::add(GPropertyMap<float>* surf)
     484 {
     485     assert(!isClosed());
     486     GPropertyMap<float>* ssurf = createStandardSurface(surf) ;
     487     addDirect(ssurf);
     488 }
     489 
     490 void GSurfaceLib::addDirect(GPropertyMap<float>* surf)
     491 {
     492     assert(!isClosed());
     493     m_surfaces.push_back(surf);
     494 }

::

     548 
     549 GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>* src)
     550 {
     551     GProperty<float>* _detect           = NULL ;
     552     GProperty<float>* _absorb           = NULL ;
     553     GProperty<float>* _reflect_specular = NULL ;
     554     GProperty<float>* _reflect_diffuse  = NULL ;

     ...
     572         if(src->isSensor())  // this means it has non-zero EFFICIENCY or detect property
     573         {
     574             GProperty<float>* _EFFICIENCY = src->getProperty(EFFICIENCY);
     575             assert(_EFFICIENCY && os && "sensor surfaces must have an efficiency" );
     576 
     577             if(m_fake_efficiency >= 0.f && m_fake_efficiency <= 1.0f)
     578             {
     579                 _detect           = makeConstantProperty(m_fake_efficiency) ;
     580                 _absorb           = makeConstantProperty(1.0-m_fake_efficiency);
     581                 _reflect_specular = makeConstantProperty(0.0);
     582                 _reflect_diffuse  = makeConstantProperty(0.0);
     583             }
     584             else
     585             {
     586                 _detect = _EFFICIENCY ;
     587                 _absorb = GProperty<float>::make_one_minus( _detect );
     588                 _reflect_specular = makeConstantProperty(0.0);
     589                 _reflect_diffuse  = makeConstantProperty(0.0);
     590             }
     591         }
     592         else
     593         {
     594             GProperty<float>* _REFLECTIVITY = src->getProperty(REFLECTIVITY);
     595             assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );
     596 

     276 template <class T>
     277 bool GPropertyMap<T>::isSensor()
     278 {
     279     return hasNonZeroProperty(EFFICIENCY) || hasNonZeroProperty(detect) ;
     280 }

     785 template <typename T>
     786 bool GPropertyMap<T>::hasNonZeroProperty(const char* pname)
     787 {
     788      if(!hasProperty(pname)) return false ;
     789      GProperty<T>* prop = getProperty(pname);
     790      return !prop->isZero();
     791 }


     40 GBorderSurface* X4LogicalBorderSurface::Convert(const G4LogicalBorderSurface* src)
     41 {
     42     const char* name = X4::Name( src );
     43     size_t index = X4::GetOpticksIndex( src ) ;
     44 
     45     G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
     46     assert( os );
     47     GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ;
     48     assert( optical_surface );
     49 
     50     GBorderSurface* dst = new GBorderSurface( name, index, optical_surface) ;
     51     // standard domain is set by GBorderSurface::init
     52 
     53     X4LogicalSurface::Convert( dst, src);
     54 
     55     const G4VPhysicalVolume* pv1 = src->GetVolume1();
     56     const G4VPhysicalVolume* pv2 = src->GetVolume2();

     34 void X4LogicalSurface::Convert(GPropertyMap<float>* dst,  const G4LogicalSurface* src)
     35 {
     36     LOG(LEVEL) << "[" ; 
     37     const G4SurfaceProperty*  psurf = src->GetSurfaceProperty() ;   
     38     const G4OpticalSurface* opsurf = dynamic_cast<const G4OpticalSurface*>(psurf);
     39     assert( opsurf );   
     40     G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable() ;
     41     X4MaterialPropertiesTable::Convert( dst, mpt );
     42     
     43     LOG(LEVEL) << "]" ;
     44 }







Darwin.blyth.1042::

    2020-12-20 19:32:44.516 INFO  [6807645] [X4PhysicalVolume::init@187] [
    2020-12-20 19:32:44.516 INFO  [6807645] [X4PhysicalVolume::init@188]  query :  queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2020-12-20 19:32:44.521 INFO  [6807645] [X4PhysicalVolume::convertMaterials@255]  num_materials 36 num_material_with_efficiency 1
    2020-12-20 19:32:44.521 INFO  [6807645] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    2020-12-20 19:32:44.522 INFO  [6807645] [X4PhysicalVolume::convertSurfaces@277] [
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@54]  NumberOfBorderSurfaces 10
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceTop
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceBot
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SSTOilSurface
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear1
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear2
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] NearIWSCurtainSurface
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] NearOWSLinerSurface
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] NearDeadLinerSurface
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf2
    2020-12-20 19:32:44.526 INFO  [6807645] [GSurfaceLib::dumpSurfaces@749] X4PhysicalVolume::convertSurfaces num_surfaces 48
     index :  0 is_sensor : N type :        bordersurface name :                                   ESRAirSurfaceTop bpv1 /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap0xc2664680x3eeae20 bpv2 /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR0xc4110d00x3eeab80 .
     index :  1 is_sensor : N type :        bordersurface name :                                   ESRAirSurfaceBot bpv1 /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap0xbfa64580x3eeb320 bpv2 /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR0xbf9bd080x3eeb080 .
     index :  2 is_sensor : N type :        bordersurface name :                                      SSTOilSurface bpv1 /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0 bpv2 /dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100 .
     index :  3 is_sensor : N type :        bordersurface name :                               SSTWaterSurfaceNear1 bpv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf5280x3efb9c0 bpv2 /dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100 .
     index :  4 is_sensor : N type :        bordersurface name :                               SSTWaterSurfaceNear2 bpv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE20xc0479c80x3efbb80 bpv2 /dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100 .
     index :  5 is_sensor : N type :        bordersurface name :                              NearIWSCurtainSurface bpv1 /dd/Geometry/Pool/lvNearPoolCurtain#pvNearPoolIWS0xc15a4980x3fa6c80 bpv2 /dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain0xc5c5f200x3fa9070 .
     index :  6 is_sensor : N type :        bordersurface name :                                NearOWSLinerSurface bpv1 /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS0xbf55b100x4128cf0 bpv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 .
     index :  7 is_sensor : N type :        bordersurface name :                               NearDeadLinerSurface bpv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 bpv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 .
     index :  8 is_sensor : Y type :        bordersurface name :                          SCB_photocathode_logsurf1 bpv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 bpv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 .
     index :  9 is_sensor : Y type :        bordersurface name :                          SCB_photocathode_logsurf2 bpv1 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 bpv2 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 .
     index : 10 is_sensor : N type :          skinsurface name :                               NearPoolCoverSurface sslv lvNearTopCover0xc1370600x3ebf2d0 .
     index : 11 is_sensor : N type :          skinsurface name :                                       RSOilSurface sslv lvRadialShieldUnit0xc3d7ec00x3eea9d0 .
     index : 12 is_sensor : N type :          skinsurface name :                                 AdCableTraySurface sslv lvAdVertiCableTray0xc3a27f00x3f2ce70 .



Darwin.charles.1062 is_sensor not set for SCB_photocathode_logsurf1 whereas is is in 1042::

    2020-12-20 20:24:14.929 INFO  [6861666] [GSurfaceLib::add@345]  GBorderSurface  name NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 keys REFLECTIVITY EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:24:14.929 INFO  [6861666] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:24:14.930 INFO  [6861666] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x118c87e80
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:24:14.931 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:24:14.931 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@86]  pns 23 pns_null 22
    2020-12-20 20:24:14.931 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@122]  cpns 33 cpns_null 33
    2020-12-20 20:24:14.931 INFO  [6861666] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:24:14.931 INFO  [6861666] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 0
    2020-12-20 20:24:14.931 INFO  [6861666] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 597.
    Process 81029 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.



Darwin.blyth.1042::


    X4PhysicalVolume=INFO X4LogicalBorderSurfaceTable=INFO X4LogicalBorderSurface=INFO GSurfaceLib=INFO X4LogicalSurface=INFO X4MaterialPropertiesTable=INFO  geocache-create -D 

    2020-12-20 20:24:45.281 INFO  [6862364] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:24:45.281 INFO  [6862364] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 1
    2020-12-20 20:24:45.281 INFO  [6862364] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:24:45.282 INFO  [6862364] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf2
    2020-12-20 20:24:45.282 INFO  [6862364] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115c74c30
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@86]  pns 23 pns_null 22
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@122]  cpns 33 cpns_null 33
    2020-12-20 20:24:45.282 INFO  [6862364] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:24:45.282 INFO  [6862364] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf2 is_sensor 1
    2020-12-20 20:24:45.282 INFO  [6862364] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf2 pv1 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 pv2 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 keys EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:24:45.283 INFO  [6862364] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :      0x115c589d0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115c58cd0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0




Darwin.charles.1062::

    2020-12-20 20:01:15.835 INFO  [6835068] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:01:15.835 INFO  [6835068] [GSurfaceLib::add@345]  GBorderSurface  name NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 keys REFLECTIVITY EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:01:15.835 INFO  [6835068] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:01:15.835 INFO  [6835068] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x118df7880
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@86]  pns 23 pns_null 22
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@122]  cpns 33 cpns_null 33
    2020-12-20 20:01:15.836 INFO  [6835068] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:01:15.836 INFO  [6835068] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 595.
    Process 77034 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.

    Process 77034 launched: '/Users/charles/local/opticks/lib/OKX4Test' (x86_64)


Possibly is_sensor is what is different.


1062 zero EFFICIENCY::

    2020-12-20 20:44:10.015 INFO  [6882139] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:44:10.015 INFO  [6882139] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x118498a30
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@87] X4MaterialPropertiesTable::AddProperties.EFFICIENCY zero  constant: 0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 22
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2020-12-20 20:44:10.016 INFO  [6882139] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:44:10.016 INFO  [6882139] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 0
    2020-12-20 20:44:10.016 INFO  [6882139] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /


1042 non-zero EFFICIENCY::

    2020-12-20 20:47:26.773 INFO  [6886110] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:47:26.773 INFO  [6886110] [*X4LogicalBorderSurface::Convert@61] NearDeadLinerSurface is_sensor 0
    2020-12-20 20:47:26.773 INFO  [6886110] [GSurfaceLib::add@345]  GBorderSurface  name NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 keys REFLECTIVITY EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:47:26.773 INFO  [6886110] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:47:26.773 INFO  [6886110] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115d22780
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@87] X4MaterialPropertiesTable::AddProperties.EFFICIENCY range: 0 : 0.24
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 22
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2020-12-20 20:47:26.774 INFO  [6886110] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:47:26.774 INFO  [6886110] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 1
    2020-12-20 20:47:26.774 INFO  [6886110] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:47:26.774 INFO  [6886110] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf2
    2020-12-20 20:47:26.774 INFO  [6886110] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115d22780
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@87] X4MaterialPropertiesTable::AddProperties.EFFICIENCY range: 0 : 0.24
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 22
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2020-12-20 20:47:26.775 INFO  [6886110] [X4LogicalSurface::Convert@43] ]




Added test CGDMLPropertyTest the 1042 vs 1062 difference is plain
-------------------------------------------------------------------

1042 has some values::

    epsilon:tests blyth$ om-;TEST=CGDMLPropertyTest;om-t 
    === om-mk : bdir /usr/local/opticks/build/cfg4/tests rdir cfg4/tests : make CGDMLPropertyTest && ./CGDMLPropertyTest
    [ 98%] Built target CFG4
    Scanning dependencies of target CGDMLPropertyTest
    [ 98%] Building CXX object tests/CMakeFiles/CGDMLPropertyTest.dir/CGDMLPropertyTest.cc.o
    [100%] Linking CXX executable CGDMLPropertyTest
    [100%] Built target CGDMLPropertyTest
    2020-12-21 12:36:59.293 INFO  [7364433] [main@36] OKConf::Geant4VersionInteger() : 1042
    2020-12-21 12:36:59.293 INFO  [7364433] [main@42]  parsing /tmp/v1.gdml
    G4GDML: Reading '/tmp/v1.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/v1.gdml' done!
    2020-12-21 12:36:59.716 INFO  [7364433] [main@47]  nmat 36
    2020-12-21 12:36:59.716 INFO  [7364433] [main@50]  nlbs 10
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] ESRAirSurfaceTop 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd97d11a0b0 plen       39
    0.98505 0.98505 0.984548 0.983585 0.968716 0.970241 0.971068 0.965398 0.951738 0.981663 0.980112 0.988567 0.985178 0.965753 0.975675 0.977987 0.975159 0.965276 0.966203 0.961376 0.958306 0.957332 0.726586 0.11559 0.104132 0.116531 0.142322 0.118947 0.179949 0.173176 0.09159 0.0100036 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd97d11a840 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] ESRAirSurfaceBot 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd97d11db90 plen       39
    0.98505 0.98505 0.984548 0.983585 0.968716 0.970241 0.971068 0.965398 0.951738 0.981663 0.980112 0.988567 0.985178 0.965753 0.975675 0.977987 0.975159 0.965276 0.966203 0.961376 0.958306 0.957332 0.726586 0.11559 0.104132 0.116531 0.142322 0.118947 0.179949 0.173176 0.09159 0.0100036 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd97d11de90 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SSTOilSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd97d128550 plen       39
    0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd97d128d20 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SSTWaterSurfaceNear1 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979c1c3a0 plen       39
    0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979c1c6a0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SSTWaterSurfaceNear2 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979c1c4d0 plen       39
    0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979c1c560 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] NearIWSCurtainSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979c300e0 plen       39
    0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.982963 0.987913 0.99 0.995 0.99 0.960524 0.940209 0.91045 0.870413 0.800692 0.760381 0.720299 0.680216 0.640134 0.600051 0.6 0.6 0.6 0.6 0.6 0.6 0.6 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979c303e0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] NearOWSLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979f853a0 plen       39
    0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.982963 0.987913 0.99 0.995 0.99 0.960524 0.940209 0.91045 0.870413 0.800692 0.760381 0.720299 0.680216 0.640134 0.600051 0.6 0.6 0.6 0.6 0.6 0.6 0.6 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f856a0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] NearDeadLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979f88930 plen       39
    0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.982963 0.987913 0.99 0.995 0.99 0.960524 0.940209 0.91045 0.870413 0.800692 0.760381 0.720299 0.680216 0.640134 0.600051 0.6 0.6 0.6 0.6 0.6 0.6 0.6 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f89100 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SCB_photocathode_logsurf1 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f8ac90 plen       39
    0.0001 0.0001 0.000440306 0.000782349 0.00112439 0.00146644 0.00180848 0.00272834 0.00438339 0.00692303 0.00998793 0.0190265 0.027468 0.0460445 0.0652553 0.0849149 0.104962 0.139298 0.170217 0.19469 0.214631 0.225015 0.24 0.235045 0.21478 0.154862 0.031507 0.00478915 0.00242326 0.000850572 0.000475524 0.000100476 7.50165e-05 5.00012e-05 2.49859e-05 0 0 0 0 
    2020-12-21 12:36:59.717 INFO  [7364433] [main@70] SCB_photocathode_logsurf2 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f8ac90 plen       39
    0.0001 0.0001 0.000440306 0.000782349 0.00112439 0.00146644 0.00180848 0.00272834 0.00438339 0.00692303 0.00998793 0.0190265 0.027468 0.0460445 0.0652553 0.0849149 0.104962 0.139298 0.170217 0.19469 0.214631 0.225015 0.24 0.235045 0.21478 0.154862 0.031507 0.00478915 0.00242326 0.000850572 0.000475524 0.000100476 7.50165e-05 5.00012e-05 2.49859e-05 0 0 0 0 
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 



1062 gives all zeros from the same GDML::

    2020-12-21 12:37:29.267 INFO  [7365045] [main@36] OKConf::Geant4VersionInteger() : 1062
    2020-12-21 12:37:29.267 INFO  [7365045] [main@42]  parsing /tmp/v1.gdml
    G4GDML: Reading '/tmp/v1.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/v1.gdml' done!
    2020-12-21 12:37:29.683 INFO  [7365045] [main@47]  nmat 36
    2020-12-21 12:37:29.683 INFO  [7365045] [main@50]  nlbs 10
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] ESRAirSurfaceTop 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] ESRAirSurfaceBot 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] SSTOilSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] SSTWaterSurfaceNear1 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] SSTWaterSurfaceNear2 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] NearIWSCurtainSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] NearOWSLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] NearDeadLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] SCB_photocathode_logsurf1 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] SCB_photocathode_logsurf2 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    epsilon:cfg4 charles$ 


See no changes to Geant4 code for property handling with::

   g4n-cls G4MaterialPropertiesTable
   g4n-cls G4PhysicsVector 

So suspicion now pointing to a GDML parsing difference. 

::

    g4n-cd source/persistency/gdml/src

    epsilon:src blyth$ g4n-cls G4GDMLReadStructure



Need a piecemeal way to test. Exploit protected methods with X4GDMLReadStructure ?::

     54 class G4GDMLReadStructure : public G4GDMLReadParamvol
     55 {
     56 
     57  public:
     58 
     59    G4GDMLReadStructure();
     60    virtual ~G4GDMLReadStructure();
     61 
     62    G4VPhysicalVolume* GetPhysvol(const G4String&) const;
     63    G4LogicalVolume* GetVolume(const G4String&) const;
     64    G4AssemblyVolume* GetAssembly(const G4String&) const;
     65    G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume*) const;
     66    G4VPhysicalVolume* GetWorldVolume(const G4String&);
     67    const G4GDMLAuxMapType* GetAuxMap() const {return &auxMap;}
     68    void Clear();   // Clears internal map and evaluator
     69 
     70    virtual void VolumeRead(const xercesc::DOMElement* const);
     71    virtual void Volume_contentRead(const xercesc::DOMElement* const);
     72    virtual void StructureRead(const xercesc::DOMElement* const);
     73 
     74  protected:
     75 
     76    void AssemblyRead(const xercesc::DOMElement* const);
     77    void DivisionvolRead(const xercesc::DOMElement* const);
     78    G4LogicalVolume* FileRead(const xercesc::DOMElement* const);
     79    void PhysvolRead(const xercesc::DOMElement* const,
     80                     G4AssemblyVolume* assembly=0);
     81    void ReplicavolRead(const xercesc::DOMElement* const, G4int number);
     82    void ReplicaRead(const xercesc::DOMElement* const replicaElement,
     83                     G4LogicalVolume* logvol,G4int number);
     84    EAxis AxisRead(const xercesc::DOMElement* const axisElement);
     85    G4double QuantityRead(const xercesc::DOMElement* const readElement);
     86    void BorderSurfaceRead(const xercesc::DOMElement* const);
     87    void SkinSurfaceRead(const xercesc::DOMElement* const);
     88 
     89  protected:
     90 
     91    G4GDMLAuxMapType auxMap;
     92    G4GDMLAssemblyMapType assemblyMap;
     93    G4LogicalVolume *pMotherLogical;
     94    std::map<std::string, G4VPhysicalVolume*> setuptoPV;
     95    G4bool strip;
     96 };




::


   ..3164     <!-- SCB manual addition start : see notes/issues/sensor-gdml-review.rst -->
     3165     <!-- see bordersurface referencing at tail of structure -->
     3166 
     3167     <opticalsurface finish="0" model="0" name="SCB_photocathode_opsurf" type="0" value="1">
     3168          <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>   <!-- the non-zero efficiency-->
     3169     </opticalsurface>
     3170     <!-- SCB manual addition end : see notes/issues/sensor-gdml-review.rst -->
     3171 
     3172   </solids>


    31971     <!-- SCB manual addition start : see notes/issues/sensor-gdml-review.rst -->
    31972     <!-- see opticalsurface at tail of solids -->
    31973 
    31974     <bordersurface name="SCB_photocathode_logsurf1" surfaceproperty="SCB_photocathode_opsurf">
    31975        <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
    31976        <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
    31977     </bordersurface>
    31978 
    31979     <bordersurface name="SCB_photocathode_logsurf2" surfaceproperty="SCB_photocathode_opsurf">
    31980        <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
    31981        <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
    31982     </bordersurface>
    31983     <!-- SCB manual addition end : see notes/issues/sensor-gdml-review.rst -->
    31984   </structure>





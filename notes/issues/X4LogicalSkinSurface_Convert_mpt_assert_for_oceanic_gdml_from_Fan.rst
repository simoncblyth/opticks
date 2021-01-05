X4LogicalSkinSurface_Convert_mpt_assert_for_oceanic_gdml_from_Fan.rst
=======================================================================


::

    epsilon:g4ok blyth$ lldb_ G4OKTest -- --gdmlpath $HOME/fan/geometry.gdml 
    (lldb) target create "G4OKTest"
    Current executable set to 'G4OKTest' (x86_64).
    (lldb) settings set -- target.run-args  "--gdmlpath" "/Users/blyth/fan/geometry.gdml"
    (lldb) r
    Process 28801 launched: '/usr/local/opticks/lib/G4OKTest' (x86_64)
    2021-01-05 14:19:52.547 INFO  [17421523] [G4Opticks::G4Opticks@305] ctor : DISABLE FPE detection : as it breaks OptiX launches

      C4FPEDetection::InvalidOperationDetection_Disable       NOT IMPLEMENTED 
    G4GDML: Reading '/Users/blyth/fan/geometry.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/Users/blyth/fan/geometry.gdml' done!
    2021-01-05 14:19:53.566 INFO  [17421523] [BOpticksKey::SetKey@90]  spec G4OKTest.X4PhysicalVolume.Water_LV0x5577b731e4b0_PV.076a017c9a406819565d369c7aa89f15
    2021-01-05 14:19:53.567 INFO  [17421523] [BOpticksResource::initViaKey@785] 
                 BOpticksKey  : KEYSOURCE
          spec (OPTICKS_KEY)  : G4OKTest.X4PhysicalVolume.Water_LV0x5577b731e4b0_PV.076a017c9a406819565d369c7aa89f15
                     exename  : G4OKTest
             current_exename  : G4OKTest
                       class  : X4PhysicalVolume
                     volname  : Water_LV0x5577b731e4b0_PV
                      digest  : 076a017c9a406819565d369c7aa89f15
                      idname  : G4OKTest_Water_LV0x5577b731e4b0_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2021-01-05 14:19:53.567 INFO  [17421523] [*G4Opticks::InitOpticks@186] 
    # BOpticksKey::export_ 
    export OPTICKS_KEY=G4OKTest.X4PhysicalVolume.Water_LV0x5577b731e4b0_PV.076a017c9a406819565d369c7aa89f15

    2021-01-05 14:19:53.567 INFO  [17421523] [*G4Opticks::InitOpticks@206] instanciate Opticks using embedded commandline only 
     --compute --embedded --xanalytic --save --natural --printenabled --pindex 0  
    2021-01-05 14:19:53.568 INFO  [17421523] [Opticks::init@437] COMPUTE_MODE compute_requested  hostname epsilon.local
    2021-01-05 14:19:53.568 INFO  [17421523] [Opticks::init@446]  mandatory keyed access to geometry, opticksaux 
    2021-01-05 14:19:53.568 INFO  [17421523] [Opticks::init@465] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER 
    2021-01-05 14:19:53.571 INFO  [17421523] [*G4Opticks::translateGeometry@766] ( GGeo instanciate
    2021-01-05 14:19:53.572 INFO  [17421523] [*G4Opticks::translateGeometry@769] ) GGeo instanciate 
    2021-01-05 14:19:53.572 INFO  [17421523] [*G4Opticks::translateGeometry@771] ( GGeo populate
    2021-01-05 14:19:53.573 ERROR [17421523] [X4MaterialTable::init@88] PROCEEDING TO convert material with no mpt Supportor_MT
    2021-01-05 14:19:53.573 INFO  [17421523] [X4PhysicalVolume::convertMaterials@255]  num_materials 6 num_material_with_efficiency 0
    2021-01-05 14:19:53.573 INFO  [17421523] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 0
    Process 28801 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x70)
        frame #0: 0x000000010598af51 libG4materials.dylib`std::__1::vector<G4String, std::__1::allocator<G4String> >::vector(std::__1::vector<G4String, std::__1::allocator<G4String> > const&) [inlined] std::__1::vector<G4String, std::__1::allocator<G4String> >::size(this=0x0000000000000068 size=0) const at vector:632
       629 	
       630 	    _LIBCPP_INLINE_VISIBILITY
       631 	    size_type size() const _NOEXCEPT
    -> 632 	        {return static_cast<size_type>(this->__end_ - this->__begin_);}
       633 	    _LIBCPP_INLINE_VISIBILITY
       634 	    size_type capacity() const _NOEXCEPT
       635 	        {return __base::capacity();}
    Target 0: (G4OKTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x70)
      * frame #0: 0x000000010598af51 libG4materials.dylib`std::__1::vector<G4String, std::__1::allocator<G4String> >::vector(std::__1::vector<G4String, std::__1::allocator<G4String> > const&) [inlined] std::__1::vector<G4String, std::__1::allocator<G4String> >::size(this=0x0000000000000068 size=0) const at vector:632
        frame #1: 0x000000010598af51 libG4materials.dylib`std::__1::vector<G4String, std::__1::allocator<G4String> >::vector(this=0x00007ffeefbfb6b0 size=0, __x=size=0) at vector:1197
        frame #2: 0x0000000105987a3d libG4materials.dylib`std::__1::vector<G4String, std::__1::allocator<G4String> >::vector(this=0x00007ffeefbfb6b0 size=0, __x=size=0) at vector:1193
        frame #3: 0x00000001059d0e60 libG4materials.dylib`G4MaterialPropertiesTable::GetMaterialPropertyNames(this=0x0000000000000000) const at G4MaterialPropertiesTable.cc:525
        frame #4: 0x00000001003cbb5f libExtG4.dylib`X4MaterialPropertiesTable::AddProperties(pmap=0x00000001140abbf0, mpt=0x0000000000000000) at X4MaterialPropertiesTable.cc:58
        frame #5: 0x00000001003cbb1f libExtG4.dylib`X4MaterialPropertiesTable::init(this=0x00007ffeefbfba80) at X4MaterialPropertiesTable.cc:49
        frame #6: 0x00000001003cbaef libExtG4.dylib`X4MaterialPropertiesTable::X4MaterialPropertiesTable(this=0x00007ffeefbfba80, pmap=0x00000001140abbf0, mpt=0x0000000000000000) at X4MaterialPropertiesTable.cc:44
        frame #7: 0x00000001003cbab5 libExtG4.dylib`X4MaterialPropertiesTable::X4MaterialPropertiesTable(this=0x00007ffeefbfba80, pmap=0x00000001140abbf0, mpt=0x0000000000000000) at X4MaterialPropertiesTable.cc:43
        frame #8: 0x00000001003cba84 libExtG4.dylib`X4MaterialPropertiesTable::Convert(pmap=0x00000001140abbf0, mpt=0x0000000000000000) at X4MaterialPropertiesTable.cc:36
        frame #9: 0x00000001003d2cc5 libExtG4.dylib`X4LogicalSurface::Convert(dst=0x00000001140abbf0, src=0x000000010e293810) at X4LogicalSurface.cc:41
        frame #10: 0x00000001003d2a44 libExtG4.dylib`X4LogicalSkinSurface::Convert(src=0x000000010e293810) at X4LogicalSkinSurface.cc:49
        frame #11: 0x00000001003d2230 libExtG4.dylib`X4LogicalSkinSurfaceTable::init(this=0x00007ffeefbfc2b8) at X4LogicalSkinSurfaceTable.cc:61
        frame #12: 0x00000001003d1f44 libExtG4.dylib`X4LogicalSkinSurfaceTable::X4LogicalSkinSurfaceTable(this=0x00007ffeefbfc2b8, dst=0x000000011408c6f0) at X4LogicalSkinSurfaceTable.cc:44
        frame #13: 0x00000001003d1efd libExtG4.dylib`X4LogicalSkinSurfaceTable::X4LogicalSkinSurfaceTable(this=0x00007ffeefbfc2b8, dst=0x000000011408c6f0) at X4LogicalSkinSurfaceTable.cc:43
        frame #14: 0x00000001003d1ecc libExtG4.dylib`X4LogicalSkinSurfaceTable::Convert(dst=0x000000011408c6f0) at X4LogicalSkinSurfaceTable.cc:36
        frame #15: 0x00000001003dfbfb libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfd638) at X4PhysicalVolume.cc:285
        frame #16: 0x00000001003df37f libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfd638) at X4PhysicalVolume.cc:192
        frame #17: 0x00000001003df065 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfd638, ggeo=0x0000000114089680, top=0x000000010e293a70) at X4PhysicalVolume.cc:177
        frame #18: 0x00000001003de325 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfd638, ggeo=0x0000000114089680, top=0x000000010e293a70) at X4PhysicalVolume.cc:168
        frame #19: 0x00000001000e3194 libG4OK.dylib`G4Opticks::translateGeometry(this=0x000000010e260be0, top=0x000000010e293a70) at G4Opticks.cc:772
        frame #20: 0x00000001000e2854 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010e260be0, world=0x000000010e293a70) at G4Opticks.cc:447
        frame #21: 0x00000001000e2665 libG4OK.dylib`G4Opticks::setGeometry(this=0x000000010e260be0, gdmlpath="/Users/blyth/fan/geometry.gdml") at G4Opticks.cc:433
        frame #22: 0x000000010000f5d2 G4OKTest`G4OKTest::initGeometry(this=0x00007ffeefbfe858) at G4OKTest.cc:190
        frame #23: 0x000000010000f022 G4OKTest`G4OKTest::init(this=0x00007ffeefbfe858) at G4OKTest.cc:144
        frame #24: 0x000000010000edc9 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe858, argc=3, argv=0x00007ffeefbfe8c0) at G4OKTest.cc:114
        frame #25: 0x000000010000f083 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe858, argc=3, argv=0x00007ffeefbfe8c0) at G4OKTest.cc:113
        frame #26: 0x0000000100011e08 G4OKTest`main(argc=3, argv=0x00007ffeefbfe8c0) at G4OKTest.cc:377
        frame #27: 0x00007fff77c24015 libdyld.dylib`start + 1
    (lldb) 


::

     265 /**
     266 X4PhysicalVolume::convertSurfaces
     267 -------------------------------------
     268 
     269 * G4LogicalSkinSurface -> GSkinSurface -> slib
     270 * G4LogicalBorderSurface -> GBorderSurface -> slib
     271 
     272 
     273 **/
     274 
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


     48 void X4LogicalSkinSurfaceTable::init()
     49 {
     50     unsigned num_src = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ;
     51     assert( num_src == m_src->size() );
     52 
     53     LOG(LEVEL) << " NumberOfSkinSurfaces num_src " << num_src ;
     54 
     55     for(size_t i=0 ; i < m_src->size() ; i++)
     56     {
     57         G4LogicalSkinSurface* src = (*m_src)[i] ;
     58 
     59         LOG(LEVEL) << src->GetName() ;
     60 
     61         GSkinSurface* dst = X4LogicalSkinSurface::Convert( src );
     62 
     63         assert( dst );
     64 
     65         m_dst->add(dst) ; // GSurfaceLib
     66     }
     67 }


     36 GSkinSurface* X4LogicalSkinSurface::Convert(const G4LogicalSkinSurface* src)
     37 {
     38     const char* name = X4::Name( src );
     39     size_t index = X4::GetOpticksIndex( src ) ;
     40 
     41     G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
     42     assert( os );
     43     GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ;
     44     assert( optical_surface );
     45 
     46     GSkinSurface* dst = new GSkinSurface( name, index, optical_surface) ;
     47     // standard domain is set by GSkinSurface::init
     48 
     49     X4LogicalSurface::Convert( dst, src);
     50 
     51     const G4LogicalVolume* lv = src->GetLogicalVolume();
     52 
     53 
     54     /*
     55     LOG(fatal) 
     56          << " X4::Name(lv)  " << X4::Name(lv)
     57          << " X4::BaseNameAsis(lv) " << X4::BaseNameAsis(lv)
     58          ; 
     59     */
     60 
     61     dst->setSkinSurface(  X4::BaseNameAsis(lv) ) ;
     62 
     63 
     64     return dst ;
     65 }


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


     34 void X4MaterialPropertiesTable::Convert( GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt )
     35 {
     36     X4MaterialPropertiesTable xtab(pmap, mpt);
     37 }
     38 
     39 X4MaterialPropertiesTable::X4MaterialPropertiesTable( GPropertyMap<float>* pmap,  const G4MaterialPropertiesTable* const mpt )
     40     :
     41     m_pmap(pmap),
     42     m_mpt(mpt)
     43 {
     44     init();
     45 }
     46 
     47 void X4MaterialPropertiesTable::init()
     48 {
     49     AddProperties( m_pmap, m_mpt );
     50 }
     51 
     52 
     53 void X4MaterialPropertiesTable::AddProperties(GPropertyMap<float>* pmap, const G4MaterialPropertiesTable* const mpt)   // static
     54 {
     55     typedef G4MaterialPropertyVector MPV ;
     56     G4bool warning ;
     57 
     58     std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ;
     59 
     60     unsigned pns_null = 0 ;
     61 
     62     for( unsigned i=0 ; i < pns.size() ; i++)
     63     {
     64         const std::string& pname = pns[i];
     65         G4int pidx = mpt->GetPropertyIndex(pname, warning=true);
     66         assert( pidx > -1 );
     67         MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning=false );
     68         LOG(LEVEL)



Problem is presumably that Opticks is assuming that the opsurf optical surface has a material properties table and it does not.


Increase verbosity::

    X4LogicalSkinSurfaceTable=INFO X4LogicalSkinSurface=INFO X4LogicalSurface=INFO X4MaterialPropertiesTable=INFO  lldb_ G4OKTest -- --gdmlpath $HOME/fan/geometry.gdml

    ...

    2021-01-05 14:36:01.484 INFO  [17439627] [Opticks::init@465] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_ANGULAR WITH_DEBUG_BUFFER WITH_WAY_BUFFER 
    2021-01-05 14:36:01.487 INFO  [17439627] [*G4Opticks::translateGeometry@766] ( GGeo instanciate
    2021-01-05 14:36:01.488 INFO  [17439627] [*G4Opticks::translateGeometry@769] ) GGeo instanciate 
    2021-01-05 14:36:01.488 INFO  [17439627] [*G4Opticks::translateGeometry@771] ( GGeo populate
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :      0x10e5047e0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :      0x10e503cb0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :      0x10e504cb0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2021-01-05 14:36:01.489 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 20
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :      0x10e505750
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :      0x10e5058c0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :      0x10e505cd0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2021-01-05 14:36:01.490 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 20
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :      0x10e5061b0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :      0x10e506af0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :      0x10e506ed0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2021-01-05 14:36:01.491 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 20
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2021-01-05 14:36:01.492 ERROR [17439627] [X4MaterialTable::init@88] PROCEEDING TO convert material with no mpt Supportor_MT
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :      0x10e507b80
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :      0x10e508550
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :      0x10e508930
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2021-01-05 14:36:01.492 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 20
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :      0x10e509260
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :      0x10e5097a0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :      0x10e509830
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :      0x10e5098f0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2021-01-05 14:36:01.493 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 19
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2021-01-05 14:36:01.494 INFO  [17439627] [X4PhysicalVolume::convertMaterials@255]  num_materials 6 num_material_with_efficiency 0
    2021-01-05 14:36:01.494 INFO  [17439627] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4LogicalSkinSurfaceTable::init@53]  NumberOfSkinSurfaces num_src 5
    2021-01-05 14:36:01.494 INFO  [17439627] [X4LogicalSkinSurfaceTable::init@59] skinSurfacePmt
    2021-01-05 14:36:01.494 INFO  [17439627] [X4LogicalSurface::Convert@36] [
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :      0x10e50a6a0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x10e50a870
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@87] X4MaterialPropertiesTable::AddProperties.EFFICIENCY range: 0.21 : 0.3
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2021-01-05 14:36:01.494 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 21
    2021-01-05 14:36:01.495 INFO  [17439627] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2021-01-05 14:36:01.495 INFO  [17439627] [X4LogicalSurface::Convert@43] ]
    2021-01-05 14:36:01.495 INFO  [17439627] [X4LogicalSkinSurfaceTable::init@59] GelSurface
    2021-01-05 14:36:01.495 INFO  [17439627] [X4LogicalSurface::Convert@36] [
    Process 29207 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x70)
        frame #0: 0x000000010598af51 libG4materials.dylib`std::__1::vector<G4String, std::__1::allocator<G4String> >::vector(std::__1::vector<G4String, std::__1::allocator<G4String> > const&) [inlined] std::__1::vector<G4String, std::__1::allocator<G4String> >::size(this=0x0000000000000068 size=0) const at vector:632
       629 	
       630 	    _LIBCPP_INLINE_VISIBILITY
       631 	    size_type size() const _NOEXCEPT
    -> 632 	        {return static_cast<size_type>(this->__end_ - this->__begin_);}
       633 	    _LIBCPP_INLINE_VISIBILITY
       634 	    size_type capacity() const _NOEXCEPT
       635 	        {return __base::capacity();}
    Target 0: (G4OKTest) stopped.
    (lldb) 


skinSurfacePmt converts OK but GelSurface does not.

::

   ...142   <solids>
      143     <sphere aunit="deg" deltaphi="360" deltatheta="15" lunit="mm" name="PMT_SV0x5577b73b10d0" rmax="149" rmin="130" startphi="0" starttheta="0"/>
      144     <opticalsurface finish="0" model="0" name="PMT_OS" type="0" value="1">
      145       <property name="REFLECTIVITY" ref="REFLECTIVITY0x5577b731cf70"/>
      146       <property name="EFFICIENCY" ref="EFFICIENCY0x5577b731d050"/>
      147     </opticalsurface>
      148     <sphere aunit="deg" deltaphi="360" deltatheta="15" lunit="mm" name="Gel0_SV0x5577b73b0230" rmax="150" rmin="130" startphi="0" starttheta="0"/>
      149     <opticalsurface finish="0" model="0" name="GelSurface" type="1" value="1"/>
      150     <box lunit="mm" name="SiPM_PV0x5577b73b1410" x="31.9248290319329" y="31.9248290319329" z="2"/>
      151     <sphere aunit="deg" deltaphi="360" deltatheta="10" lunit="mm" name="Gel1_SV0x5577b73b03a0" rmax="150" rmin="130" startphi="0" starttheta="0"/>
      152     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="Supportor_SV0x5577b73aff00" rmax="150" rmin="0" startphi="0" starttheta="0"/>
      153     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="DOM_SV0x5577b7328cb0" rmax="165" rmin="0" startphi="0" starttheta="0"/>
      154     <opticalsurface finish="0" model="0" name="DomGlassSurface" type="1" value="1"/>
      155     <box lunit="mm" name="Water_SV0x5577b731e430" x="1000000" y="1000000" z="1000000"/>
      156   </solids>



    16369     <skinsurface name="skinSurfacePmt" surfaceproperty="PMT_OS">
    16370       <volumeref ref="PMT_LV0x5577b73b1240"/>
    16371     </skinsurface>
    16372     <skinsurface name="GelSurface" surfaceproperty="GelSurface">
    16373       <volumeref ref="Gel0_LV0x5577b73b0510"/>
    16374     </skinsurface>
    16375     <skinsurface name="skinSurfaceSipm" surfaceproperty="PMT_OS">
    16376       <volumeref ref="SiPM_LV0x5577b73b1490"/>
    16377     </skinsurface>
    16378     <skinsurface name="GelSurface" surfaceproperty="GelSurface">
    16379       <volumeref ref="Gel1_LV0x5577b73b05c0"/>
    16380     </skinsurface>
    16381     <skinsurface name="DomGlassSurface" surfaceproperty="DomGlassSurface">
    16382       <volumeref ref="DOM_LV0x5577b7328e20"/>
    16383     </skinsurface>
    16384   </structure>



The opticalsurface "GelSurface" without any properties fails in the conversion. 


Have improved the error handling, making an assert get tripped sooner::

    2021-01-05 14:49:31.087 INFO  [17453056] [X4MaterialPropertiesTable::AddProperties@96]  pns 23 pns_null 21
    2021-01-05 14:49:31.087 INFO  [17453056] [X4MaterialPropertiesTable::AddProperties@132]  cpns 33 cpns_null 33
    2021-01-05 14:49:31.087 INFO  [17453056] [X4LogicalSurface::Convert@43] ]
    2021-01-05 14:49:31.087 INFO  [17453056] [X4LogicalSkinSurfaceTable::init@59] GelSurface
    2021-01-05 14:49:31.087 INFO  [17453056] [X4LogicalSurface::Convert@36] [
    2021-01-05 14:49:31.087 FATAL [17453056] [X4MaterialPropertiesTable::Convert@37] cannot convert a null G4MaterialPropertiesTable : this usually means you have omitted to setup any properties for a surface or material
    Assertion failed: (mpt), function Convert, file /Users/blyth/opticks/extg4/X4MaterialPropertiesTable.cc, line 38.
    Process 29941 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (G4OKTest) stopped.
    (lldb) 





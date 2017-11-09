Surface Review Test Geometry
=================================

Problem with surfaces in test geometry
----------------------------------------

Cause of confusion is the conflation of two things:

* surface properties
* surface location (specified by sslv/bpv1/bpv2)

With test geometry wish to reuse some surface properties, 
from the base geometry but need to totally change surface location
to suit the boundary spec coming down the pipe (from NCSGList).

So need to derive a separate GSurfaceLib from scratch that is able to 
draw from the basis one for surface properties. This implies 
creating a from scratch GBndLib too.

Added some methods to GSurfaceLib to allow passing props from basis into 
a new lib with different locations::   

     89         // methods to assist with de-conflation of surface props and location
     90         void addBorderSurface(GPropertyMap<float>* surf, const char* pv1, const char* pv2);
     91         void addSkinSurface(GPropertyMap<float>* surf, const char* sslv_ );



tboolean-media
----------------

::

    simon:ggeo blyth$ tboolean-;tboolean-media -D --okg4 --dbgsurf --dbgbnd 

Hmm need to get GSurLib outa the picture...::

    2017-11-09 20:07:42.054 FATAL [4038096] [CGeometry::init@61] CGeometry::init G4 simple test geometry 
    2017-11-09 20:07:42.054 INFO  [4038096] [GSurLib::init@66] [--dbgsurf] GSurLib::init
    2017-11-09 20:07:42.054 INFO  [4038096] [GSurLib::init@71] [--dbgsurf] GSurLib::init m_bordersurface.size 8
    2017-11-09 20:07:42.054 INFO  [4038096] [GSurLib::collectSur@107] [--dbgsurf] m_slib numSurfaces 1
    2017-11-09 20:07:42.054 INFO  [4038096] [GSurLib::collectSur@119] [--dbgsurf] i   0 type S name perfectAbsorbSurface
    2017-11-09 20:07:42.054 INFO  [4038096] [CPropLib::init@66] CPropLib::init
    2017-11-09 20:07:42.054 INFO  [4038096] [CPropLib::initCheckConstants@118] CPropLib::initCheckConstants mm 1 MeV 1 nanosecond 1 ns 1 nm 1e-06 GC::nanometer 1e-06 h_Planck 4.13567e-12 GC::h_Planck 4.13567e-12 c_light 299.792 GC::c_light 299.792 dscale 0.00123984
    2017-11-09 20:07:42.054 INFO  [4038096] [*CTestDetector::makeDetector_NCSG@172] CTestDetector::makeDetector_NCSG numSolids 1
    2017-11-09 20:07:42.054 INFO  [4038096] [*NCSGList::createUniverse@155] NCSGList::createUniverse bnd0 Rock//perfectAbsorbSurface/Pyrex ubnd Rock///Rock scale 1 delta 1
    2017-11-09 20:07:42.055 INFO  [4038096] [GPropertyLib::getIndex@366] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname []
    2017-11-09 20:07:42.056 INFO  [4038096] [GPropertyLib::close@410] GPropertyLib::close type GSurfaceLib buf 48,2,39,4
    2017-11-09 20:07:42.056 INFO  [4038096] [GBndLib::addBoundary@329] [--dbgbnd]  spec Rock///Rock flip 0 bnd  ( 11,---,---, 11) boundary 11
    2017-11-09 20:07:42.057 FATAL [4038096] [*CTestDetector::makeChildVolume@142]  csg.spec Rock///Rock boundary 11 mother - lv UniverseLV_box pv UniversePV_box mat Rock
    2017-11-09 20:07:42.057 INFO  [4038096] [GBndLib::addBoundary@329] [--dbgbnd]  spec Rock//perfectAbsorbSurface/Pyrex flip 0 bnd  ( 11,---, 45, 13) boundary 123
    2017-11-09 20:07:42.057 FATAL [4038096] [*CTestDetector::makeChildVolume@142]  csg.spec Rock//perfectAbsorbSurface/Pyrex boundary 123 mother UniverseLV_box lv box_lv0_ pv box_pv0_ mat Pyrex
    2017-11-09 20:07:42.057 INFO  [4038096] [CDetector::traverse@103] [--dbgsurf] CDetector::traverse START 
    2017-11-09 20:07:42.057 INFO  [4038096] [CTraverser::Traverse@135] [--dbgsurf] CTraverser::Traverse START 
    2017-11-09 20:07:42.057 INFO  [4038096] [CTraverser::Traverse@142] [--dbgsurf] CTraverser::Traverse DONE numSelected 2 bbox NBoundingBox low -401.0000,-401.0000,-401.0000 high 401.0000,401.0000,401.0000 ce 0.0000,0.0000,0.0000,401.0000 pvs.size 2 lvs.size 2
    2017-11-09 20:07:42.057 INFO  [4038096] [CTraverser::Summary@111] CDetector::traverse numMaterials 2 numMaterialsWithoutMPT 0
    2017-11-09 20:07:42.057 INFO  [4038096] [CDetector::traverse@114] [--dbgsurf] CDetector::traverse DONE 
    2017-11-09 20:07:42.057 INFO  [4038096] [CDetector::attachSurfaces@294] [--dbgsurf] CDetector::attachSurfaces START closing gsurlib, creating csurlib  
    2017-11-09 20:07:42.057 INFO  [4038096] [CSurfaceLib::convert@81] [--dbgsurf] CSurfaceLib::convert  num_surf 48
    2017-11-09 20:07:42.057 INFO  [4038096] [*CSurfaceLib::makeSkinSurface@189] CSurfaceLib::makeSkinSurface name                NearPoolCoverSurface lvn                NearPoolCoverSurface lv NULL
    2017-11-09 20:07:42.057 INFO  [4038096] [*CSurfaceLib::makeBorderSurface@156] CSurfaceLib::makeBorderSurface name NearDeadLinerSurface bpv1 __dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018 bpv2 __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270 pvn1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c018 pvn2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b270
    2017-11-09 20:07:42.057 INFO  [4038096] [*CTraverser::getPV@312] CTraverser::getPV name /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c018 index -1 num_indices 0
    2017-11-09 20:07:42.057 FATAL [4038096] [*CTraverser::getPV@325] CTraverser::getPV name /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c018 index -1 m_pvs 2 m_pvnames 2
    Assertion failed: (valid), function getPV, file /Users/blyth/opticks/cfg4/CTraverser.cc, line 332.
    Process 77555 stopped
    * thread #1: tid = 0x3d9dd0, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) 

    (lldb) bt
    * thread #1: tid = 0x3d9dd0, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001043ad145 libcfg4.dylib`CTraverser::getPV(this=0x00000001110a9ea0, name=0x00000001110ad240) + 1701 at CTraverser.cc:332
        frame #5: 0x00000001043cd784 libcfg4.dylib`CDetector::getPV(this=0x0000000111099d60, name=0x00000001110ad240) + 36 at CDetector.cc:229
        frame #6: 0x00000001043dabb8 libcfg4.dylib`CSurfaceLib::makeBorderSurface(this=0x000000011109a280, surf=0x000000010a11c7e0, os=0x00000001110ac810) + 1352 at CSurfaceLib.cc:164
        frame #7: 0x00000001043d9d05 libcfg4.dylib`CSurfaceLib::convert(this=0x000000011109a280, detector=0x0000000111099d60) + 709 at CSurfaceLib.cc:90
        frame #8: 0x00000001043ce459 libcfg4.dylib`CDetector::attachSurfaces(this=0x0000000111099d60) + 265 at CDetector.cc:297
        frame #9: 0x0000000104346bba libcfg4.dylib`CGeometry::init(this=0x0000000111099cf0) + 762 at CGeometry.cc:73
        frame #10: 0x00000001043468b0 libcfg4.dylib`CGeometry::CGeometry(this=0x0000000111099cf0, hub=0x0000000109640ca0) + 112 at CGeometry.cc:53
        frame #11: 0x0000000104346c1d libcfg4.dylib`CGeometry::CGeometry(this=0x0000000111099cf0, hub=0x0000000109640ca0) + 29 at CGeometry.cc:54
        frame #12: 0x00000001043f6d86 libcfg4.dylib`CG4::CG4(this=0x000000010be5ea50, hub=0x0000000109640ca0) + 214 at CG4.cc:120
        frame #13: 0x00000001043f731d libcfg4.dylib`CG4::CG4(this=0x000000010be5ea50, hub=0x0000000109640ca0) + 29 at CG4.cc:142
        frame #14: 0x00000001044f1cc3 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe430, argc=29, argv=0x00007fff5fbfe510) + 547 at OKG4Mgr.cc:35
        frame #15: 0x00000001044f1f53 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe430, argc=29, argv=0x00007fff5fbfe510) + 35 at OKG4Mgr.cc:41
        frame #16: 0x00000001000132ee OKG4Test`main(argc=29, argv=0x00007fff5fbfe510) + 1486 at OKG4Test.cc:56
        frame #17: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 


CDetector is grabbing the wrong GSurfaceLib ::

     m_slib(new CSurfaceLib(m_hub->getSurfaceLib())), 




GGeoTest::relocateSurfacesBoundarySetup
----------------------------------------

* trying to reuse surf props from basis lib, basic testing in OpticksHubTest 

* relocation complications 

  * may change metadata surftype (which is ctor argument)
  * possible metadata key mixage  
  * bordersurface on top volumes leaves placeholder UNIVERSE_PV for CTestDetector 
    to replace with name of the universe wrapper

::

    230 void GGeoTest::relocateSurfacesBoundarySetup(GSolid* solid, const char* spec)
    231 {
    232     BBnd b(spec);
    233     bool unknown_osur = b.osur && !m_slib->hasSurface(b.osur) ;
    234     bool unknown_isur = b.isur && !m_slib->hasSurface(b.isur) ;
    235 
    236     if(unknown_osur || unknown_isur)
    237     {
    238         GSolid* parent = static_cast<GSolid*>(solid->getParent()) ;
    239         const char* self_lv = solid->getLVName() ;
    240         const char* self_pv = solid->getPVName() ;
    241         const char* parent_pv = parent ? parent->getPVName() : UNIVERSE_PV ;
    242 
    243         if(m_dbgbnd)
    244         LOG(error)
    245               << "[--dbgbnd]"
    246               << " spec " << spec
    247               << " unknown_osur " << unknown_osur
    248               << " unknown_isur " << unknown_isur
    249               << " self_lv " << self_lv
    250               << " self_pv " << self_pv
    251               << " parent_pv " << parent_pv
    252               ;
    253 
    254         if( b.osur == b.isur ) // skin 
    255         {
    256             m_slib->relocateBasisSkinSurface( b.osur, self_lv );
    257         }
    258         else if( b.isur ) // border self->parent
    259         {
    260             m_slib->relocateBasisBorderSurface( b.isur, self_pv, parent_pv  );
    261         }
    262         else if( b.osur ) // border parent->self
    263         {
    264             m_slib->relocateBasisBorderSurface( b.osur, parent_pv, self_pv ) ;
    265         }
    266     } 
    267 
    268     unsigned boundary = m_bndlib->addBoundary(spec, false);  // only adds if not existing
    269     solid->setBoundary(boundary);     // unlike ctor these create arrays
    270 }


::

    704 void GSurfaceLib::importForTex2d()
    705 {
    706     unsigned int ni = m_buffer->getShape(0); // surfaces
    707     unsigned int nj = m_buffer->getShape(1); // payload categories 
    708     unsigned int nk = m_buffer->getShape(2); // wavelength samples
    709     unsigned int nl = m_buffer->getShape(3); // 4 props
    710 
    711     assert(m_standard_domain->getLength() == nk );
    712 
    713     float* data = m_buffer->getValues();
    714 
    715     for(unsigned int i=0 ; i < ni ; i++)
    716     {
    717         const char* key = m_names->getKey(i);
    718 
    719         LOG(debug) << std::setw(3) << i
    720                    << " " << key ;
    721 
    722         GOpticalSurface* os = NULL ;
    723 
    724         NMeta* surfmeta = m_meta ? m_meta->getObj(key) : NULL  ;
    725 
    726         const char* surftype = AssignSurfaceType(surfmeta) ;
    727 
    728         GPropertyMap<float>* surf = new GPropertyMap<float>(key,i, surftype, os, surfmeta );
    729 
    730         for(unsigned int j=0 ; j < nj ; j++)
    731         {
    732             import(surf, data + i*nj*nk*nl + j*nk*nl , nk, nl, j );
    733         }
    734 




How to handle test geometry in CSurfaceLib::convert ?
------------------------------------------------------------

**Best way** 
    prepare the GSurfaceLib in a manner such that CSurfaceLib 
    doesnt need to know if test/full geometry.


GGeoTest : GMaterialLib from base + AbInitio GBndLib/GSurfaceLib  
---------------------------------------------------------------------

* dev in GBndLibInitTest 
* How to handle surface indices in the bndlib ? GBndLib buffers are dynamic to handle added surfaces, so may just work ?

* from OptiX point of view (GPU geometry) all thats needed is the GBndLib to create the texture 

::
 
    141 GSolid* GMaker::makeFromCSG(NCSG* csg, GBndLib* bndlib, unsigned verbosity )
    142 {
    ...
    160     GSolid* solid = new GSolid(index, transform, mesh, UINT_MAX, NULL );
    161 
    162     // csg is mesh-qty not a node-qty, boundary spec is a node-qty : so this is just for testing
    163 
    164     unsigned boundary = bndlib->addBoundary(spec);  // only adds if not existing
    165 
    166     solid->setBoundary(boundary);     // unlike ctor these create arrays
    167 
    168     solid->setSensor( NULL );
    169 
    170 
    171     OpticksCSG_t type = csg->getRootType() ;
    172 
    173     const char* shapename = CSGName(type);
    174     std::string lvn = GMaker::LVName(shapename, index);
    175     std::string pvn = GMaker::PVName(shapename, index);
    176 
    177     solid->setPVName( strdup(pvn.c_str()) );
    178     solid->setLVName( strdup(lvn.c_str()) );
    179     solid->setCSGFlag( type );
    180 
    181     GParts* pts = GParts::make( csg, spec, verbosity );
    182 
    183 
    184     solid->setParts( pts );


::

    simon:opticks blyth$ opticks-find GGeoTest | grep new
    ./ggeo/GGeoTest.cc:    m_config(new GGeoTestConfig(ok->getTestConfig())),
    ./ggeo/tests/GGeoTestConfigTest.cc:    GGeoTestConfig* gtc = new GGeoTestConfig(CONFIG);
    ./opticksgeo/OpticksHub.cc:    GGeoTest* testgeo = new GGeoTest(m_ok, basis);
    simon:opticks blyth$ 


::

    295 void OpticksHub::loadGeometry()
    296 {
    297     assert(m_geometry == NULL && "OpticksHub::loadGeometry should only be called once");
    298 
    299     LOG(info) << "OpticksHub::loadGeometry START" ;
    300 
    301     m_geometry = new OpticksGeometry(this);   // m_lookup is set into m_ggeo here 
    302 
    303     m_geometry->loadGeometry();
    304 
    305 
    306     //   Lookup A and B are now set ...
    307     //      A : by OpticksHub::configureLookupA (ChromaMaterialMap.json)
    308     //      B : on GGeo loading in GGeo::setupLookup
    309 
    310     m_ggeo = m_geometry->getGGeo();
    311     m_gscene = m_ggeo->getScene();
    312 
    313     if(m_ok->isTest())
    314     {
    315         LOG(info) << "OpticksHub::loadGeometry --test modifying geometry" ;
    316 
    317         assert(m_geotest == NULL);
    318 
    319         GGeoBase* basis = getGGeoBase(); // ana OR tri depending on --gltf
    320 
    321         m_geotest = createTestGeometry(basis);
    322     }
    323     else


    339 GGeoTest* OpticksHub::createTestGeometry(GGeoBase* basis)
    340 {
    341     assert(m_ok->isTest());
    342 
    343     LOG(info) << "OpticksHub::createTestGeometry START" ;
    344 
    345     GGeoTest* testgeo = new GGeoTest(m_ok, basis);
    346 
    347     LOG(info) << "OpticksHub::createTestGeometry DONE" ;
    348 
    349     return testgeo ;
    350 }





* GMaker::makeFromCSG assigns PV, LV names to solids
* GGeoTest collects solids into GNodeLib 





[RESOLVED] Containing Material placeholder leading to DUD in buffer
=====================================================================

Resolved by change order of *App::prepareOptiX* and configuring the analytic PMT 
containing material.

Issue
---------

Boundary issues with::

    ggv --analyticmesh 1 


Planting an assert to see where the GMaterialLib CLOSE is coming from::

    2016-Mar-14 17:08:30.793589]:info: OGeo::convert nmm 2
    [2016-Mar-14 17:08:30.793673]:info: GGeoLib::getMergedMesh index 0 m_ggeo 0x1073080a0 mm 0x11029af50 meshverbosity 0
    [2016-Mar-14 17:08:30.798398]:info: OGeo::makeTriangulatedGeometry  mmIndex 0 numFaces (PrimitiveCount) 434816 numSolids 12230 numITransforms 1
    [2016-Mar-14 17:08:30.798546]:info: GMesh::makeFaceRepeatedInstancedIdentityBuffer numSolids 12230 numFaces (sum of faces in numSolids)434816 numITransforms 1 numRepeatedIdentity 434816
    [2016-Mar-14 17:08:30.804055]:info: OGeo::makeTriangulatedGeometry using FaceRepeatedInstancedIdentityBuffer friid items 434816 numITransforms*numFaces 434816
    [2016-Mar-14 17:08:30.808273]:info: OGeo::makeMaterial  radiance_ray 1 propagate_ray 0
    [2016-Mar-14 17:08:30.815576]:info: GGeoLib::getMergedMesh index 1 m_ggeo 0x1073080a0 mm 0x11029b8c0 meshverbosity 0
    [2016-Mar-14 17:08:30.815708]:info: OGeo::makeRepeatedGroup numTransforms 672 numIdentity 3360 numSolids 5
    OGeo::makeRepeatedGroup islice NSlice      0 :   672 :     1  
    [2016-Mar-14 17:08:30.815858]:info: GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [CONTAINING_MATERIAL]
    Assertion failed: (0), function getIndex, file /Users/blyth/env/optix/ggeo/GPropertyLib.cc, line 116.
    Process 15242 stopped
    * thread #1: tid = 0x4a561e, 0x00007fff8a219866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8a219866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8a219866:  jae    0x7fff8a219870            ; __pthread_kill + 20
       0x7fff8a219868:  movq   %rax, %rdi
       0x7fff8a21986b:  jmp    0x7fff8a216175            ; cerror_nocancel
       0x7fff8a219870:  retq   
    (lldb) bt
    * thread #1: tid = 0x4a561e, 0x00007fff8a219866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8a219866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff818b635c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff88606b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff885d09bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001022b98e0 libGGeo.dylib`GPropertyLib::getIndex(this=0x000000011218b9d0, shortname=0x0000000120f95f61) + 848 at GPropertyLib.cc:116
        frame #5: 0x00000001022a92a7 libGGeo.dylib`GBndLib::parse(this=0x00000001102a1980, spec=0x00000001122485f0, flip=false) + 2199 at GBndLib.cc:146
        frame #6: 0x00000001022a982e libGGeo.dylib`GBndLib::addBoundary(this=0x00000001102a1980, spec=0x00000001122485f0, flip=false) + 62 at GBndLib.cc:162
        frame #7: 0x000000010237be0a libGGeo.dylib`GParts::registerBoundaries(this=0x0000000112248080) + 346 at GParts.cc:198
        frame #8: 0x000000010237bc99 libGGeo.dylib`GParts::close(this=0x0000000112248080) + 25 at GParts.cc:186
        frame #9: 0x000000010248f006 libOptiXRap.dylib`OGeo::makeAnalyticGeometry(this=0x0000000120f5f760, mm=0x000000011029b8c0) + 470 at OGeo.cc:433
        frame #10: 0x000000010248bee5 libOptiXRap.dylib`OGeo::makeGeometry(this=0x0000000120f5f760, mergedmesh=0x000000011029b8c0) + 261 at OGeo.cc:411
        frame #11: 0x000000010248cec2 libOptiXRap.dylib`OGeo::makeRepeatedGroup(this=0x0000000120f5f760, mm=0x000000011029b8c0) + 1170 at OGeo.cc:233
        frame #12: 0x000000010248b7bc libOptiXRap.dylib`OGeo::convert(this=0x0000000120f5f760) + 1612 at OGeo.cc:170
        frame #13: 0x0000000103828596 libGGeoViewLib.dylib`App::prepareOptiX(this=0x00007fff5fbfe070) + 3158 at App.cc:740
        frame #14: 0x000000010000b4ea GGeoView`main(argc=4, argv=0x00007fff5fbfe218) + 682 at main.cc:38
        frame #15: 0x00007fff8568c5fd libdyld.dylib`start + 1
        frame #16: 0x00007fff8568c5fd libdyld.dylib`start + 1


    (lldb) f 7
    frame #7: 0x000000010237be0a libGGeo.dylib`GParts::registerBoundaries(this=0x0000000112248080) + 346 at GParts.cc:198
       195     for(unsigned int i=0 ; i < nbnd ; i++)
       196     {
       197         const char* spec = m_bndspec->getKey(i);
    -> 198         unsigned int boundary = m_bndlib->addBoundary(spec);
       199         setBoundary(i, boundary);
       200  
       201         if(m_verbose)
    (lldb) p spec
    (const char *) $0 = 0x00000001122485f0 "CONTAINING_MATERIAL///Pyrex"
    (lldb) 

    (lldb) f 6
    frame #6: 0x00000001022a982e libGGeo.dylib`GBndLib::addBoundary(this=0x00000001102a1980, spec=0x00000001122485f0, flip=false) + 62 at GBndLib.cc:162
       159  
       160  unsigned int GBndLib::addBoundary( const char* spec, bool flip)
       161  {
    -> 162      guint4 bnd = parse(spec, flip);
       163      add(bnd);
       164      return index(bnd) ; 
       165  }
    (lldb) f 5
    frame #5: 0x00000001022a92a7 libGGeo.dylib`GBndLib::parse(this=0x00000001102a1980, spec=0x00000001122485f0, flip=false) + 2199 at GBndLib.cc:146
       143      const char* isur_ = elem[2].c_str() ;
       144      const char* imat_ = elem[3].c_str() ;
       145  
    -> 146      unsigned int omat = m_mlib->getIndex(omat_) ;
       147      unsigned int osur = m_slib->getIndex(osur_) ;
       148      unsigned int isur = m_slib->getIndex(isur_) ;
       149      unsigned int imat = m_mlib->getIndex(imat_) ;
    (lldb) p omat_
    (const char *) $1 = 0x0000000120f95f61 "CONTAINING_MATERIAL"
    (lldb) 




The original 123 boundaries are increased by dynamic ones from the analytic PMT, but 
it seems that the bounds do not reflect that.
Are making the bnd buffer too soon ?

::

    In [1]: 123*4
    Out[1]: 492

    In [2]: 128*4
    Out[2]: 512


    [2016-Mar-14 16:58:47.831485]:info: App:: prelaunch
    [2016-Mar-14 16:58:47.831580]:info: OContext::launch entry 1 width 500000 height 1
    wavelength_lookup OUT OF BOUNDS nm   413.8957 nmi    18.1948 line  492 offset    0 boundary_bounds (   0,  38,   0, 491) boundary_domain (   60.0000,  820.0000,   20.0000,  760.0000) 
    wavelength_lookup OUT OF BOUNDS nm   413.8957 nmi    18.1948 line  495 offset    0 boundary_bounds (   0,  38,   0, 491) boundary_domain (   60.0000,  820.0000,   20.0000,  760.0000) 
    wavelength_lookup OUT OF BOUNDS nm   448.2245 nmi    19.9112 line  492 offset    0 boundary_bounds (   0,  38,   0, 491) boundary_domain (   60.0000,  820.0000,   20.0000,  760.0000) 
    wavelength_lookup OUT OF BOUNDS nm   413.8957 nmi    18.1948 line  493 offset    0 boundary_bounds (   0,  38,   0, 491) boundary_domain (   60.0000,  820.0000,   20.0000,  760.0000) 
    wavelength_lookup OUT OF BOUNDS nm   417.6102 nmi    18.3805 line  492 offset    0 boundary_bounds (   0,  38,   0, 491) boundary_domain (   60.0000,  820.0000,   20.0000,  760.0000) 
    wavelength_lookup OUT OF BOUNDS nm   434.8695 nmi    19.2435 line  492 offset    0 boundary_bounds (   0,  38,   0, 491) boundary_domain (   60.0000,  820.0000,   20.0000,  760.0000) 
    wavelength_lookup OUT OF BOUNDS nm   419.6159 nmi    18.4808 line  492 offset    0 boundary_bounds (   0,  38,   0, 491) boundary_domain (   60.0000,  820.0000,   20.0000,  760.0000) 


The replace is omitted::

    simon:env blyth$ find . -name '*.*' -type f -exec grep -H CONTAINING_MATERIAL {} \;
    ./nuwa/detdesc/pmt/dd.py:    def __init__(self, top="CONTAINING_MATERIAL", sensor="SENSOR_SURFACE"):
    ./optix/ggeo/GParts.cc:const char* GParts::CONTAINING_MATERIAL = "CONTAINING_MATERIAL" ;  
    ./optix/ggeo/GParts.cc:    m_bndspec->replaceField(0, GParts::CONTAINING_MATERIAL, material );
    ./optix/ggeo/GParts.hh:       static const char* CONTAINING_MATERIAL ; 


::

    168 void GParts::setContainingMaterial(const char* material)
    169 {
    170     // for flexibility persisted GParts should leave the outer containing material
    171     // set to a default marker name, to allow the GParts to be placed within other geometry
    172 
    173     m_bndspec->replaceField(0, GParts::CONTAINING_MATERIAL, material );
    174 }
    175 
    176 void GParts::setSensorSurface(const char* surface)
    177 {
    178     m_bndspec->replaceField(1, GParts::SENSOR_SURFACE, surface ) ;
    179     m_bndspec->replaceField(2, GParts::SENSOR_SURFACE, surface ) ;
    180 }


::

    simon:env blyth$ find . -name '*.*' -type f -exec grep -H setContainingMaterial {} \;
    ./optix/ggeo/GGeoTest.cc:    analytic->setContainingMaterial(container_inner_material);    // match outer material of PMT with inner material of the box
    ./optix/ggeo/GParts.cc:void GParts::setContainingMaterial(const char* material)
    ./optix/ggeo/GParts.hh:        void setContainingMaterial(const char* material="MineralOil");


Setup done by GGeoTest::createPmtInBox but not with full geometry::

    144 GMergedMesh* GGeoTest::createPmtInBox()
    145 {
    146     // somewhat dirtily associates analytic geometry with triangulated for the PMT 
    147     //
    148     //   * detdesc parsed analytic geometry in GPmt (see pmt-ecd dd.py tree.py etc..)
    149     //   * instance-1 GMergedMesh 
    150     //
    151     // using prior DYB specific knowledge...
    152     // mergedMesh-repeat-candidate-1 is the triangulated PMT 5-solids 
    153     //
    154     // assumes single container 
    155 
    ...
    180     GMergedMesh* triangulated = GMergedMesh::combine( mmpmt->getIndex(), mmpmt, solids );
    ...
    185 
    186     GParts* analytic = triangulated->getParts();
    187     analytic->setContainingMaterial(container_inner_material);    // match outer material of PMT with inner material of the box
    188     analytic->setSensorSurface("lvPmtHemiCathodeSensorSurface") ; // kludge, TODO: investigate where triangulated gets this from
    189     analytic->close();
    190 
    191     // needed by OGeo::makeAnalyticGeometry
    192 
    193     NPY<unsigned int>* idBuf = mmpmt->getAnalyticInstancedIdentityBuffer();
    194     NPY<float>* itransforms = mmpmt->getITransformsBuffer();
    195 
    196     assert(idBuf);
    197     assert(itransforms);
    198 
    199     triangulated->setAnalyticInstancedIdentityBuffer(idBuf);
    200     triangulated->setITransformsBuffer(itransforms);
    201 
    202     return triangulated ;
    203 }





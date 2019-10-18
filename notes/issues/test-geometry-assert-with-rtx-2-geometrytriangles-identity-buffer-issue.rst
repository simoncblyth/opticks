test-geometry-assert-with-rtx-2-geometrytriangles-identity-buffer-issue
==============================================================================

Context : scan-ph-13
-------------------------


Fixed by distinguising between input analytic form and output form which can be tri in GGeoTest 
---------------------------------------------------------------------------------------------------

Vis, shows a red raytrace::

   ts box --xtriangle



Issue
--------

* suspect lack of an identity buffer with test geometry 


::

    [blyth@localhost opticks]$ scan-ph
                   scan-- : ts box --pfx scan-ph-13 --cat cvd_1_rtx_2_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 3 --cvd 1 --rtx 2 --xtriangle ======= RC 139  RC 0x8b 
                   scan-- : ts box --pfx scan-ph-13 --cat cvd_1_rtx_2_10M --generateoverride 10000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 10 --cvd 1 --rtx 2 --xtriangle ======= RC 139  RC 0x8b 
                   scan-- : ts box --pfx scan-ph-13 --cat cvd_1_rtx_2_100M --generateoverride 100000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 100 --cvd 1 --rtx 2 --xtriangle ======= RC 139  RC 0x8b 
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ 
    [blyth@localhost opticks]$ ts box --pfx scan-ph-13 --cat cvd_1_rtx_2_1M --generateoverride 1000000 --compute --production --savehit --multievent 10 --xanalytic --nog4propagate --rngmax 3 --cvd 1 --rtx 2 --xtriangle -D


::

    2019-10-18 15:13:16.786 INFO  [219762] [OGeo::init@212] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2019-10-18 15:13:16.786 INFO  [219762] [GGeoLib::dump@366] OGeo::convert GGeoLib ANALYTIC  numMergedMesh 1 ptr 0x5b6c240
    mm index   0 geocode   A                  numVolumes          2 numFaces        4408 numITransforms           2 numITransforms*numVolumes           4 GParts Y GPts Y
     num_total_volumes 2 num_instanced_volumes 0 num_global_volumes 2
       0 pts Y  GPts.NumPt 2 lvIdx ( 0 1)
    2019-10-18 15:13:16.787 INFO  [219762] [OGeo::convert@238] [ nmm 1
    2019-10-18 15:13:16.787 ERROR [219762] [Opticks::isXAnalytic@1035]  --xanalytic option overridded by --xtriangle  
    
    Program received signal SIGSEGV, Segmentation fault.
    0x00007ffff7551a44 in std::vector<int, std::allocator<int> >::size (this=0x8) at /usr/include/c++/4.8.2/bits/stl_vector.h:646
    646       { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007ffff7551a44 in std::vector<int, std::allocator<int> >::size (this=0x8) at /usr/include/c++/4.8.2/bits/stl_vector.h:646
    #1  0x00007fffe995b8ca in NPYBase::getNumItems (this=0x0, ifr=0, ito=1) at /home/blyth/opticks/npy/NPYBase.cpp:352
    #2  0x00007fffeab87d39 in GMesh::makeFaceRepeatedInstancedIdentityBuffer (this=0x5f85370) at /home/blyth/opticks/ggeo/GMesh.cc:2097
    #3  0x00007fffeab88950 in GMesh::getFaceRepeatedInstancedIdentityBuffer (this=0x5f85370) at /home/blyth/opticks/ggeo/GMesh.cc:2219
    #4  0x00007fffeab88621 in GMesh::getAppropriateRepeatedIdentityBuffer (this=0x5f85370) at /home/blyth/opticks/ggeo/GMesh.cc:2196
    #5  0x00007ffff6554231 in OGeo::makeGeometryTriangles (this=0x7d50ad0, mm=0x5f85370, lod=0) at /home/blyth/opticks/optixrap/OGeo.cc:805
    #6  0x00007ffff65523ad in OGeo::makeOGeometry (this=0x7d50ad0, mergedmesh=0x5f85370, lod=0) at /home/blyth/opticks/optixrap/OGeo.cc:577
    #7  0x00007ffff65508ae in OGeo::makeGlobalGeometryGroup (this=0x7d50ad0, mm=0x5f85370) at /home/blyth/opticks/optixrap/OGeo.cc:298
    #8  0x00007ffff65504fb in OGeo::convertMergedMesh (this=0x7d50ad0, i=0) at /home/blyth/opticks/optixrap/OGeo.cc:277
    #9  0x00007ffff6550013 in OGeo::convert (this=0x7d50ad0) at /home/blyth/opticks/optixrap/OGeo.cc:244
    #10 0x00007ffff6547c01 in OScene::init (this=0x73d0fe0) at /home/blyth/opticks/optixrap/OScene.cc:169
    #11 0x00007ffff65474d7 in OScene::OScene (this=0x73d0fe0, hub=0x6f26a0, cmake_target=0x7ffff68fe36c "OptiXRap", ptxrel=0x0) at /home/blyth/opticks/optixrap/OScene.cc:91
    #12 0x00007ffff68a1c6f in OpEngine::OpEngine (this=0x73f3720, hub=0x6f26a0) at /home/blyth/opticks/okop/OpEngine.cc:75
    #13 0x00007ffff79cb8bb in OKPropagator::OKPropagator (this=0x742c040, hub=0x6f26a0, idx=0x621b630, viz=0x0) at /home/blyth/opticks/ok/OKPropagator.cc:68
    #14 0x00007ffff7bd45f8 in OKG4Mgr::OKG4Mgr (this=0x7fffffffc930, argc=50, argv=0x7fffffffcc78) at /home/blyth/opticks/okg4/OKG4Mgr.cc:110
    #15 0x000000000040399a in main (argc=50, argv=0x7fffffffcc78) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:27
    (gdb) f 2
    #2  0x00007fffeab87d39 in GMesh::makeFaceRepeatedInstancedIdentityBuffer (this=0x5f85370) at /home/blyth/opticks/ggeo/GMesh.cc:2097
    2097        bool iidentity_ok = m_iidentity_buffer->getNumItems() == numVolumes*numITransforms ;
    (gdb) f 5
    #5  0x00007ffff6554231 in OGeo::makeGeometryTriangles (this=0x7d50ad0, mm=0x5f85370, lod=0) at /home/blyth/opticks/optixrap/OGeo.cc:805
    805     optix::Buffer identityBuffer = createInputBuffer<optix::uint4>( mm->getAppropriateRepeatedIdentityBuffer(), RT_FORMAT_UNSIGNED_INT4, 1 , "identityBuffer"); 
    (gdb) f 6
    #6  0x00007ffff65523ad in OGeo::makeOGeometry (this=0x7d50ad0, mergedmesh=0x5f85370, lod=0) at /home/blyth/opticks/optixrap/OGeo.cc:577
    577         ogeom->gt = makeGeometryTriangles(mergedmesh, lod);
    (gdb) 


::

    (gdb) f 4
    #4  0x00007fffeab88621 in GMesh::getAppropriateRepeatedIdentityBuffer (this=0x5f85370) at /home/blyth/opticks/ggeo/GMesh.cc:2196
    2196            id = mm->getFaceRepeatedInstancedIdentityBuffer(); 
    (gdb) list
    2191        unsigned numFaces = mm->getNumFaces();
    2192    
    2193        GBuffer* id = NULL ;  
    2194        if(numITransforms > 0)  //  formerly 0   : HUH: perhaps should be 1,  always using friid even for globals ?
    2195        {
    2196            id = mm->getFaceRepeatedInstancedIdentityBuffer(); 
    2197            assert(id);
    2198            LOG(verbose) << "using FaceRepeatedInstancedIdentityBuffer" << " friid items " << id->getNumItems() << " numITransforms*numFaces " << numITransforms*numFaces ;     
    2199            assert( id->getNumItems() == numITransforms*numFaces );
    2200        }
    (gdb) p numITransforms
    $4 = 2
    (gdb) p numFaces
    $5 = 4408
    (gdb) 

::

    2019-10-18 15:36:51.539 INFO  [271320] [GGeoLib::dump@366] OGeo::convert GGeoLib ANALYTIC  numMergedMesh 1 ptr 0x5b6c250
    mm index   0 geocode   A                  numVolumes          2 numFaces        4408 numITransforms           2 numITransforms*numVolumes           4 GParts Y GPts Y
     num_total_volumes 2 num_instanced_volumes 0 num_global_volumes 2



::

    157 void GGeoTest::init()
    158 {
    159     LOG(LEVEL) << "[" ;
    160 
    161     assert( m_config->isNCSG() );
    162 
    163     GMergedMesh* tmm_ = initCreateCSG() ;
    164 
    165     if(!tmm_)
    166     {
    167         setErr(101) ;
    168         return ;
    169     }
    170 
    171     GMergedMesh* tmm = m_lod > 0 ? GMergedMesh::MakeLODComposite(tmm_, m_lodconfig->levels ) : tmm_ ;
    172 
    173     char geocode =  m_analytic ? OpticksConst::GEOCODE_ANALYTIC : OpticksConst::GEOCODE_TRIANGULATED ;  // message to OGeo
    174 
    175     assert( m_analytic ) ;
    176 
    177     tmm->setGeoCode( geocode );
    178 
    179     if(tmm->isTriangulated())
    180     {
    181         tmm->setITransformsBuffer(NULL); // avoiding FaceRepeated complications 
    182     }
    183 
    184 
    185     m_geolib->setMergedMesh( 0, tmm );  // TODO: create via standard GGeoLib::create ?
    186 
    187     LOG(LEVEL) << "]" ;
    188 }
    189 


* hmm GGeoTest normally is always analytic 

::

      56 bool GMergedMesh::isSkip() const
      57 {
      58    return m_geocode == OpticksConst::GEOCODE_SKIP ;
      59 }
      60 bool GMergedMesh::isAnalytic() const
      61 {
      62    return m_geocode == OpticksConst::GEOCODE_ANALYTIC ;
      63 }
      64 bool GMergedMesh::isTriangulated() const
      65 {
      66    return m_geocode == OpticksConst::GEOCODE_TRIANGULATED ;
      67 }



::

     636 char GMesh::getGeoCode() const
     637 {
     638     return m_geocode ;
     639 }
     640 void GMesh::setGeoCode(char geocode)
     641 {
     642     m_geocode = geocode ;
     643 }



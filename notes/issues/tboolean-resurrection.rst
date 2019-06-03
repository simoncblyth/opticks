tboolean-resurrection
===========================


::

   [blyth@localhost issues]$ tboolean-;tboolean-box --okg4 --debugger

    ...



Issue 1 : nlohmann json sticter about needs string/number in the json : FIXED AT INPUT PY STRING LEVEL
-----------------------------------------------------------------------------------------------------------

* edited the tboolean-box-- bash function python as indicated by bactraces

::

   [blyth@localhost issues]$ tboolean-;tboolean-box --okg4 --debugger
    ...
    2019-06-01 19:09:05.823 INFO  [456280] [NCSGList::load@133] NCSGList::load VERBOSITY 0 basedir /tmp/blyth/location/tboolean-box-- txtpath /tmp/blyth/location/tboolean-box--/csg.txt nbnd 2
    terminate called after throwing an instance of 'std::domain_error'
      what():  type must be number, but is string
    
    Program received signal SIGABRT, Aborted.
    0x00007fffe2035207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2035207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20368f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe29447d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffe2942746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffe2942773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffe2942993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007fffe523045a in nlohmann::basic_json<std::map, std::vector, std::string, bool, long, unsigned long, double, std::allocator>::get_impl<float, 0> (this=0x18c5318)
        at /home/blyth/local/opticks/externals/include/YoctoGL/ext/json.hpp:2711
    #7  0x00007fffe522fd6d in nlohmann::basic_json<std::map, std::vector, std::string, bool, long, unsigned long, double, std::allocator>::get<float, 0> (this=0x18c5318)
        at /home/blyth/local/opticks/externals/include/YoctoGL/ext/json.hpp:2882
    #8  0x00007fffe522eeee in NMeta::get<float> (this=0x18c4b10, name=0x7fffe526d83e "containerscale", fallback=0x7fffe526d83b "2.") at /home/blyth/opticks/npy/NMeta.cpp:207
    #9  0x00007fffe50b5638 in NPYMeta::getValue<float> (this=0x18bed10, key=0x7fffe526d83e "containerscale", fallback=0x7fffe526d83b "2.", item=-1) at /home/blyth/opticks/npy/NPYMeta.cpp:70
    #10 0x00007fffe5175539 in NCSG::postload (this=0x18b38b0) at /home/blyth/opticks/npy/NCSG.cpp:250
    #11 0x00007fffe5175297 in NCSG::loadsrc (this=0x18b38b0) at /home/blyth/opticks/npy/NCSG.cpp:233
    #12 0x00007fffe5174871 in NCSG::Load (treedir=0x18af3d8 "/tmp/blyth/location/tboolean-box--/1", config=0x18c4be0) at /home/blyth/opticks/npy/NCSG.cpp:83
    #13 0x00007fffe517456b in NCSG::Load (treedir=0x18af3d8 "/tmp/blyth/location/tboolean-box--/1") at /home/blyth/opticks/npy/NCSG.cpp:53
    #14 0x00007fffe517feda in NCSGList::loadTree (this=0x18b1a70, idx=1, boundary=0x18c4e98 "Vacuum///GlassSchottF2") at /home/blyth/opticks/npy/NCSGList.cpp:254
    #15 0x00007fffe517f8a1 in NCSGList::load (this=0x18b1a70) at /home/blyth/opticks/npy/NCSGList.cpp:156
    #16 0x00007fffe517f03b in NCSGList::Load (csgpath=0x18b7530 "/tmp/blyth/location/tboolean-box--", verbosity=0, checkmaterial=true) at /home/blyth/opticks/npy/NCSGList.cpp:40
    #17 0x00007fffe5cda40e in GGeoTest::GGeoTest (this=0x18aeed0, ok=0x69f680, basis=0x6cff00) at /home/blyth/opticks/ggeo/GGeoTest.cc:122
    #18 0x00007fffe71012e5 in OpticksHub::createTestGeometry (this=0x6b8420, basis=0x6cff00) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:562
    #19 0x00007fffe7100d3d in OpticksHub::loadGeometry (this=0x6b8420) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:514
    #20 0x00007fffe70ff664 in OpticksHub::init (this=0x6b8420) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:229
    #21 0x00007fffe70ff380 in OpticksHub::OpticksHub (this=0x6b8420, ok=0x69f680) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    #22 0x00007ffff7bd51ad in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc00, argc=33, argv=0x7fffffffcf38) at /home/blyth/opticks/okg4/OKG4Mgr.cc:71
    #23 0x0000000000403998 in main (argc=33, argv=0x7fffffffcf38) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) 
    
   
::

    [blyth@localhost 0]$ jsn.py meta.json
    {u'container': u'1',
     u'containerscale': u'3',
     u'ctrl': u'0',
     u'emit': -1,
     u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55',
     u'poly': u'IM',
     u'resolution': u'20',
     u'verbosity': u'0'}
    [blyth@localhost 0]$ 



Issue 2 : hookupSD lv assert : FIXED
----------------------------------------

* loads DYB geometry, for the materials
* but also gets the cachemeta lv2sd association : need to distinguish a basis load from a normal one
  (maybe just simple as isTest ?)

* FIX: dont set m_lv2sd when using --test


::


   [blyth@localhost issues]$ tboolean-;tboolean-box --okg4 --debugger
    ...
    2019-06-01 19:22:25.192 ERROR [20168] [GGeo::loadCacheMeta@759] /home/blyth/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/cachemeta.json
    2019-06-01 19:22:25.192 INFO  [20168] [NMeta::dump@128] GGeo::loadCacheMeta
    {
        "answer": 42,
        "argline": " OKTest --gltf 3 -G",
        "lv2sd": {
            "/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98": "SD0",
            "/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca0": "SD0"
        },
        "question": "huh?"
    }
    2019-06-01 19:22:25.192 INFO  [20168] [NMeta::dump@128] GGeo::loadCacheMeta.m_lv2sd
    {
        "/dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98": "SD0",
        "/dd/Geometry/PMT/lvPmtHemiCathode0xc2cdca0": "SD0"
    }

    ...

    2019-06-01 19:22:25.443 INFO  [20168] [CSurfaceLib::convert@136] CSurfaceLib  numBorderSurface 1 numSkinSurface 0
    2019-06-01 19:22:25.443 INFO  [20168] [CDetector::attachSurfaces@340] ]
    2019-06-01 19:22:25.443 ERROR [20168] [CDetector::hookupSD@151] SetSensitiveDetector lvn /dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98 sdn SD0 lv 0
    2019-06-01 19:22:25.443 FATAL [20168] [CDetector::hookupSD@158]  no lv /dd/Geometry/PMT/lvHeadonPmtCathode0xc2c8d98
    OKG4Test: /home/blyth/opticks/cfg4/CDetector.cc:159: void CDetector::hookupSD(): Assertion `lv' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe2035207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2035207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20368f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe202e026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe202e0d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdd1c01 in CDetector::hookupSD (this=0x24a5160) at /home/blyth/opticks/cfg4/CDetector.cc:159
    #5  0x00007fffefddee84 in CTestDetector::init (this=0x24a5160) at /home/blyth/opticks/cfg4/CTestDetector.cc:84
    #6  0x00007fffefddeb7c in CTestDetector::CTestDetector (this=0x24a5160, hub=0x6b8420, query=0x0, sd=0x24a2b00) at /home/blyth/opticks/cfg4/CTestDetector.cc:59
    #7  0x00007fffefd7c663 in CGeometry::init (this=0x24a50b0) at /home/blyth/opticks/cfg4/CGeometry.cc:70
    #8  0x00007fffefd7c554 in CGeometry::CGeometry (this=0x24a50b0, hub=0x6b8420, sd=0x24a2b00) at /home/blyth/opticks/cfg4/CGeometry.cc:60
    #9  0x00007fffefdec4d3 in CG4::CG4 (this=0x22c2af0, hub=0x6b8420) at /home/blyth/opticks/cfg4/CG4.cc:121
    #10 0x00007ffff7bd5256 in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc00, argc=33, argv=0x7fffffffcf38) at /home/blyth/opticks/okg4/OKG4Mgr.cc:76
    #11 0x0000000000403998 in main (argc=33, argv=0x7fffffffcf38) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
        (gdb) 
    
    GG
    i

Try::

    [blyth@localhost ggeo]$ hg diff GGeo.cc
    diff -r db8916913476 ggeo/GGeo.cc
    --- a/ggeo/GGeo.cc  Fri May 31 22:30:55 2019 +0800
    +++ b/ggeo/GGeo.cc  Sat Jun 01 19:40:48 2019 +0800
    @@ -764,18 +764,31 @@
         {
             m_loadedcachemeta->dump("GGeo::loadCacheMeta");  
         }
    -    m_lv2sd = m_loadedcachemeta->getObj("lv2sd"); 
    +    NMeta* lv2sd = m_loadedcachemeta->getObj("lv2sd"); 
     
    -    if( m_lv2sd )
    +
    +    if( lv2sd )
         {
    -        m_lv2sd->dump("GGeo::loadCacheMeta.m_lv2sd"); 
    +        lv2sd->dump("GGeo::loadCacheMeta.lv2sd"); 
         }
         else
         {
    -        LOG(error) << " NULL m_lv2sd " ;  
    +        LOG(error) << " NULL lv2sd " ;  
         }
     
     
    +    if( m_ok->isTest() )
    +    {
    +         LOG(error) << "NOT USING the lv2sd association as --test is active " ;  
    +    }
    +    else
    +    {
    +         m_lv2sd = lv2sd ;  
    +    }
    +
    +
    +
    +
     }




Issue 3 : eui assert : expecting event UserInfo set by eg CGenstepSource : FIXED
------------------------------------------------------------------------------------

FIX : Added CEventInfo gencode to CTorchSource and CInputPhotonSource


::

   [blyth@localhost issues]$ tboolean-;tboolean-box --okg4 --debugger
   ...
   
    2019-06-01 19:43:08.738 INFO  [54035] [CWriter::initEvent@75] CWriter::initEvent dynamic STATIC(GPU style) _record_max 100000 _bounce_max  9 _steps_per_photon 10 num_g4event 10
    2019-06-01 19:43:08.738 INFO  [54035] [CRec::initEvent@87] CRec::initEvent note recstp
    2019-06-01 19:43:08.738 INFO  [54035] [CG4::propagate@330]  calling BeamOn numG4Evt 10
    2019-06-01 19:43:10.110 INFO  [54035] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2019-06-01 19:43:10.110 INFO  [54035] [CInputPhotonSource::GeneratePrimaryVertex@150] CInputPhotonSource::GeneratePrimaryVertex num_photons 10000
    2019-06-01 19:43:10.123 INFO  [54035] [CSensitiveDetector::Initialize@81]  HCE 0x988e5b0 HCE.Capacity 2 SensitiveDetectorName SD0 collectionName[0] OpHitCollectionA collectionName[1] OpHitCollectionB
    OKG4Test: /home/blyth/opticks/cfg4/CG4Ctx.cc:185: void CG4Ctx::setEvent(const G4Event*): Assertion `eui && "expecting event UserInfo set by eg CGenstepSource "' failed.
    
    Program received signal SIGABRT, Aborted.
    0x00007fffe2035207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libX11-devel-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libdrm-2.4.91-3.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2035207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20368f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe202e026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe202e0d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefdea88d in CG4Ctx::setEvent (this=0x22c2ba8, event=0x96dc890) at /home/blyth/opticks/cfg4/CG4Ctx.cc:185
    #5  0x00007fffefde70bf in CEventAction::setEvent (this=0x24fc5c0, event=0x96dc890) at /home/blyth/opticks/cfg4/CEventAction.cc:48
    #6  0x00007fffefde7087 in CEventAction::BeginOfEventAction (this=0x24fc5c0, anEvent=0x96dc890) at /home/blyth/opticks/cfg4/CEventAction.cc:39
    #7  0x00007fffec3afb40 in G4EventManager::DoProcessing (this=0x243d910, anEvent=0x96dc890) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:149
    #8  0x00007fffec3b0572 in G4EventManager::ProcessOneEvent (this=0x243d910, anEvent=0x96dc890) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:338
    #9  0x00007fffec6b2665 in G4RunManager::ProcessOneEvent (this=0x22c28f0, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:399
    #10 0x00007fffec6b24d7 in G4RunManager::DoEventLoop (this=0x22c28f0, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #11 0x00007fffec6b1d2d in G4RunManager::BeamOn (this=0x22c28f0, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #12 0x00007fffefdede13 in CG4::propagate (this=0x22c2b80) at /home/blyth/opticks/cfg4/CG4.cc:331
    #13 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffcc10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:177
    #14 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcc10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #15 0x00000000004039a7 in main (argc=33, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 
    
    
    
::

    175 void CG4Ctx::setEvent(const G4Event* event) // invoked by CEventAction::setEvent
    176 {
    177     _event = const_cast<G4Event*>(event) ;
    178     _event_id = event->GetEventID() ;
    179 
    180     _event_total += 1 ;
    181     _event_track_count = 0 ;
    182 
    183 
    184     CEventInfo* eui = (CEventInfo*)event->GetUserInformation();
    185     assert(eui && "expecting event UserInfo set by eg CGenstepSource ");
    186 
    187     _gen = eui->gencode ;
    188 
    189     LOG(info)
    190         << " gen " << _gen
    191         << " SourceType " << OpticksFlags::SourceType(_gen)
    192         ;
    193 
    194     assert( _gen == TORCH || _gen == G4GUN || _gen == CERENKOV || _gen == SCINTILLATION );
    195 }
       

    
Issue 4 : crazy placement transform is messing them all up : FIXED 
---------------------------------------------------------------------

FIXED : uninitialized m_transform bug in NTrianglesNPY 


::

    2019-06-01 20:31:49.545 INFO  [136875] [nmat4triple::dump@839] GParts::applyPlacementTransform gtransform:
     tvq 
      triple.t  0.000   0.000   0.000   0.000 
              427.500 427.500 450.000 382.500 
              450.000 427.500 382.500 427.500 
              450.000 337.500 450.000 427.500 
    np.fromstring("6.39588e-38 0 6.39588e-38 0 427.5 427.5 450 382.5 450 427.5 382.5 427.5 450 337.5 450 427.5 ", dtype=np.float32, sep=" ").reshape(4,4) 

      triple.v   -nan    -nan    -nan    -nan 
                 -nan    -nan    -nan    -nan 
                 -nan    -nan    -nan    -nan 
                 -nan    -nan    -nan    -nan 
    np.fromstring("-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan ", dtype=np.float32, sep=" ").reshape(4,4) 

      triple.q   -nan    -nan    -nan    -nan 
                 -nan    -nan    -nan    -nan 
                 -nan    -nan    -nan    -nan 
                 -nan    -nan    -nan    -nan 
    np.fromstring("-nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan -nan ", dtype=np.float32, sep=" ").reshape(4,4) 


Call stack of applyPlacementTransform with test geometry::

    (gdb) bt
    #0  0x00007ffff44f649b in raise () from /lib64/libpthread.so.0
    #1  0x00007fffe5cb219a in GParts::applyPlacementTransform (this=0x1ad57a0, gtransform=0x1bc6ba0, verbosity=3) at /home/blyth/opticks/ggeo/GParts.cc:706
    #2  0x00007fffe5cf3418 in GMergedMesh::mergeVolumeAnalytic (this=0x1e705f0, pts=0x1ad57a0, transform=0x1bc6ba0, verbosity=3) at /home/blyth/opticks/ggeo/GMergedMesh.cc:667
    #3  0x00007fffe5cf1ab1 in GMergedMesh::mergeVolume (this=0x1e705f0, volume=0x1bc6c20, selected=true, verbosity=3) at /home/blyth/opticks/ggeo/GMergedMesh.cc:429
    #4  0x00007fffe5cf063b in GMergedMesh::combine (index=0, mm=0x0, volumes=std::vector of length 2, capacity 2 = {...}, verbosity=3) at /home/blyth/opticks/ggeo/GMergedMesh.cc:156
    #5  0x00007fffe5cdc5a3 in GGeoTest::combineVolumes (this=0x18b7cf0, volumes=std::vector of length 2, capacity 2 = {...}, mm0=0x0) at /home/blyth/opticks/ggeo/GGeoTest.cc:600
    #6  0x00007fffe5cda60a in GGeoTest::initCreateCSG (this=0x18b7cf0) at /home/blyth/opticks/ggeo/GGeoTest.cc:202
    #7  0x00007fffe5cda15c in GGeoTest::init (this=0x18b7cf0) at /home/blyth/opticks/ggeo/GGeoTest.cc:137
    #8  0x00007fffe5cd9fb0 in GGeoTest::GGeoTest (this=0x18b7cf0, ok=0x69f680, basis=0x6cff00) at /home/blyth/opticks/ggeo/GGeoTest.cc:128
    #9  0x00007fffe71012e5 in OpticksHub::createTestGeometry (this=0x6b8420, basis=0x6cff00) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:562
    #10 0x00007fffe7100d3d in OpticksHub::loadGeometry (this=0x6b8420) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:514
    #11 0x00007fffe70ff664 in OpticksHub::init (this=0x6b8420) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:229
    #12 0x00007fffe70ff380 in OpticksHub::OpticksHub (this=0x6b8420, ok=0x69f680) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    #13 0x00007ffff7bd51ad in OKG4Mgr::OKG4Mgr (this=0x7fffffffcc00, argc=33, argv=0x7fffffffcf38) at /home/blyth/opticks/okg4/OKG4Mgr.cc:71
    #14 0x0000000000403998 in main (argc=33, argv=0x7fffffffcf38) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:8
    (gdb) 



Transforms on both volumes look gibberish::

    gdb) f 2
#2  0x00007fffe5cf3418 in GMergedMesh::mergeVolumeAnalytic (this=0x1e705f0, pts=0x1ad57a0, transform=0x1bc6ba0, verbosity=3) at /home/blyth/opticks/ggeo/GMergedMesh.cc:667
    667         pts->applyPlacementTransform(transform, verbosity );
    (gdb) f 3
#3  0x00007fffe5cf1ab1 in GMergedMesh::mergeVolume (this=0x1e705f0, volume=0x1bc6c20, selected=true, verbosity=3) at /home/blyth/opticks/ggeo/GMergedMesh.cc:429
    429         mergeVolumeAnalytic( pts, transform, verbosity );
    (gdb) p base
    $3 = (GNode \*) 0x0
    (gdb) p volume
    $4 = (GVolume \*) 0x1bc6c20
    (gdb) f 4
#4  0x00007fffe5cf063b in GMergedMesh::combine (index=0, mm=0x0, volumes=std::vector of length 2, capacity 2 = {...}, verbosity=3) at /home/blyth/opticks/ggeo/GMergedMesh.cc:156
    156     for(unsigned i=0 ; i < numVolumes ; i++) com->mergeVolume(volumes[i], true, verbosity ) ;
    (gdb) p volumes
    $7 = std::vector of length 2, capacity 2 = {0x1bc6c20, 0x1bc77e0}
    (gdb) p volumes[0]
    $8 = (__gnu_cxx::__alloc_traits<std::allocator<GVolume*> >::value_type &) @0x1bc8490: 0x1bc6c20
    (gdb) p volumes[0]->getTransform()
    $9 = (GMatrixF \*) 0x1bc6ba0
    (gdb) p \*volumes[0]->getTransform()
    $10 = {<GBuffer> = {m_nbytes = 64, m_pointer = 0x1e71820, m_itemsize = 4, m_nelem = 1, m_name = 0x1bc6670 "GMatrix4", m_buffer_id = -1, m_buffer_target = 0, m_bufspec = 0x0}, 
      _vptr.GMatrix = 0x7fffe5f81430 <vtable for GMatrix<float>+16>, a1 = 6.39588492e-38, a2 = 427.500122, a3 = 450, a4 = 450, b1 = 0, b2 = 427.500122, b3 = 427.500122, b4 = 337.500122, 
      c1 = 6.39588492e-38, c2 = 450, c3 = 382.500122, c4 = 450, d1 = 0, d2 = 382.500122, d3 = 427.500122, d4 = 427.500122}
    (gdb) p \*volumes[1]->getTransform()
    $11 = {<GBuffer> = {m_nbytes = 64, m_pointer = 0x0, m_itemsize = 4, m_nelem = 1, m_name = 0x1bc7230 "GMatrix4", m_buffer_id = -1, m_buffer_target = 0, m_bufspec = 0x0}, 
      _vptr.GMatrix = 0x7fffe5f81430 <vtable for GMatrix<float>+16>, a1 = 8.43719788e-38, a2 = 60.0000458, a3 = 20.0000153, a4 = 150, b1 = 0, b2 = 6.66667175, b3 = 150, b4 = 33.3333588, 
      c1 = 8.43719788e-38, c2 = 150, c3 = 46.6667023, c4 = 20.0000153, d1 = 0, d2 = 46.6667023, d3 = 6.66667175, d4 = 150}
    (gdb) 



 
Issue 4:  crazy pol values causing overflows : KLUDGED IT TO GIVE ZEROS FOR OVERFLOWS  : kludge removed, NOW FIXED PROPERLY
--------------------------------------------------------------------------------------------------------------------------------

* problem of the 1st point (and 2nd point?, trying to skip slot zero didnt work) 

::

    #9  0x00007fffe4c3ec57 in BConverter::round_to_even<unsigned char, float> (x=@0x7fffffffa37c: 2117.82422) at /home/blyth/opticks/boostrap/BConverter.cc:12
    #10 0x00007fffe4c3e9af in BConverter::my__float2uint_rn (fv=2117.82422) at /home/blyth/opticks/boostrap/BConverter.cc:43
    #11 0x00007fffefdd0036 in CWriter::writeStepPoint_ (this=0x24fc630, point=0x244f670, photon=...) at /home/blyth/opticks/cfg4/CWriter.cc:205
    #12 0x00007fffefdcfb4f in CWriter::writeStepPoint (this=0x24fc630, point=0x244f670, flag=4096, material=2) at /home/blyth/opticks/cfg4/CWriter.cc:133
    #13 0x00007fffefdc6ee5 in CRecorder::RecordStepPoint (this=0x24fc3f0, point=0x244f670, flag=4096, material=2, boundary_status=Ds::Undefined) at /home/blyth/opticks/cfg4/CRecorder.cc:485
    #14 0x00007fffefdc6863 in CRecorder::postTrackWriteSteps (this=0x24fc3f0) at /home/blyth/opticks/cfg4/CRecorder.cc:415
    #15 0x00007fffefdc5b20 in CRecorder::postTrack (this=0x24fc3f0) at /home/blyth/opticks/cfg4/CRecorder.cc:133
    #16 0x00007fffefded7aa in CG4::postTrack (this=0x22c2b10) at /home/blyth/opticks/cfg4/CG4.cc:255
    #17 0x00007fffefde96fe in CTrackingAction::PostUserTrackingAction (this=0x24fc540, track=0x762be50) at /home/blyth/opticks/cfg4/CTrackingAction.cc:91
    #18 0x00007fffec137326 in G4TrackingManager::ProcessOneTrack (this=0x243d930, apValueG4Track=0x762be50)
        at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/tracking/src/G4TrackingManager.cc:140
    #19 0x00007fffec3afd46 in G4EventManager::DoProcessing (this=0x243d8a0, anEvent=0x6d8dc20) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:185
    #20 0x00007fffec3b0572 in G4EventManager::ProcessOneEvent (this=0x243d8a0, anEvent=0x6d8dc20) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4EventManager.cc:338
    #21 0x00007fffec6b2665 in G4RunManager::ProcessOneEvent (this=0x22c2880, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:399
    #22 0x00007fffec6b24d7 in G4RunManager::DoEventLoop (this=0x22c2880, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #23 0x00007fffec6b1d2d in G4RunManager::BeamOn (this=0x22c2880, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #24 0x00007fffefdee1b5 in CG4::propagate (this=0x22c2b10) at /home/blyth/opticks/cfg4/CG4.cc:331
    #25 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffcc10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:177
    #26 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcc10) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #27 0x00000000004039a7 in main (argc=33, argv=0x7fffffffcf48) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) f 11




    (gdb) f 11
#11 0x00007fffefdd0036 in CWriter::writeStepPoint_ (this=0x24fc630, point=0x244f670, photon=...) at /home/blyth/opticks/cfg4/CWriter.cc:205
    205     unsigned char polx = BConverter::my__float2uint_rn( (pol.x()+1.f)*127.f );
    (gdb) p pol
    $5 = (const G4ThreeVector &) @0x244f6f0: {dx = 15.67578125, dy = 31.030364990234375, dz = -449.89999389648438, static tolerance = 2.22045e-14}
    (gdb) 

::

    In [1]: (15.67578125 + 1.)*127.
    Out[1]: 2117.82421875


Checking in BConverterTest, input to BConverter::my__float2uint_rn should be in range 0-255.  clearly pol.x() should be in 0-1

pol same as pos::

    gdb) f 13
    #13 0x00007fffefdc5ee5 in CRecorder::RecordStepPoint (this=0x24fbb10, point=0x244ed90, flag=4096, material=2, boundary_status=Ds::Undefined) at /home/blyth/opticks/cfg4/CRecorder.cc:485
    485     return m_writer->writeStepPoint( point, flag, material );
    (gdb) f 14
    #14 0x00007fffefdc5863 in CRecorder::postTrackWriteSteps (this=0x24fbb10) at /home/blyth/opticks/cfg4/CRecorder.cc:415
    415             done = RecordStepPoint( pre , preFlag,  u_premat,  prior_boundary_status, PRE );   
    (gdb) p pre
    $1 = (const G4StepPoint *) 0x244ed90
    (gdb) p *pre
    $2 = {fPosition = {dx = 15.67578125, dy = 31.030364990234375, dz = -449.89999389648438, static tolerance = 2.22045e-14}, fGlobalTime = 0.20000000298023224, fLocalTime = 0, fProperTime = 0, fMomentumDirection = {dx = -0, dy = -0, dz = 1, 
        static tolerance = 2.22045e-14}, fKineticEnergy = 3.262741777421046e-06, fVelocity = 299.79244995117188, fpTouchable = {fObj = 0x244e070}, fpMaterial = 0x24e6870, fpMaterialCutsCouple = 0x3acf540, fpSensitiveDetector = 0x0, fSafety = 0, 
      fPolarization = {dx = 15.67578125, dy = 31.030364990234375, dz = -449.89999389648438, static tolerance = 2.22045e-14}, fStepStatus = fUndefined, fpProcessDefinedStep = 0x0, fMass = 0, fCharge = 0, fMagneticMoment = 0, fWeight = 1}
    (gdb) 
    

::

    2019-06-01 22:44:54.218 INFO  [370063] [DsG4OpBoundaryProcess::PostStepDoIt@245]  event_id       9 photon_id       7 step_id    0  
    2019-06-01 22:44:54.218 INFO  [370063] [DsG4OpBoundaryProcess::PostStepDoIt@245]  event_id       9 photon_id       7 step_id    1  
    2019-06-01 22:44:54.218 INFO  [370063] [DsG4OpBoundaryProcess::PostStepDoIt@245]  event_id       9 photon_id       7 step_id    2  
    2019-06-01 22:44:54.218 INFO  [370063] [CWriter::writeStepPoint_@190]  pos (9.27609,-36.4964,-449.9)
    2019-06-01 22:44:54.218 INFO  [370063] [CWriter::writeStepPoint_@191]  pol (9.27609,-36.4964,-449.9)
    2019-06-01 22:44:54.218 INFO  [370063] [CWriter::writeStepPoint_@190]  pos (9.27609,-36.4964,-100)
    2019-06-01 22:44:54.218 INFO  [370063] [CWriter::writeStepPoint_@191]  pol (0.0205463,-0.0808385,-0.996515)
    2019-06-01 22:44:54.219 INFO  [370063] [CWriter::writeStepPoint_@190]  pos (9.27609,-36.4964,100)
    2019-06-01 22:44:54.219 INFO  [370063] [CWriter::writeStepPoint_@191]  pol (0.0205463,-0.0808385,-0.996515)
    2019-06-01 22:44:54.219 INFO  [370063] [CWriter::writeStepPoint_@190]  pos (9.27609,-36.4964,450)
    2019-06-01 22:44:54.219 INFO  [370063] [CWriter::writeStepPoint_@191]  pol (0.0205463,-0.0808385,-0.996515)




Issue 5 : interop downloadHits : ye olde chestnut : usual workaround succeeded : CAN VIZ THE PROPAGATION
------------------------------------------------------------------------------------------------------------------

::

    2019-06-01 22:49:36.639 INFO  [377734] [OpIndexer::indexSequenceInterop@252] OpIndexer::indexSequenceInterop slicing (OBufBase*)m_seq 
    2019-06-01 22:49:36.664 INFO  [377734] [OpEngine::propagate@132] ]
    2019-06-01 22:49:36.665 INFO  [377734] [OpticksViz::indexPresentationPrep@391] OpticksViz::indexPresentationPrep
    2019-06-01 22:49:36.666 INFO  [377734] [OpticksViz::downloadEvent@381] OpticksViz::downloadEvent (1)
    2019-06-01 22:49:36.720 INFO  [377734] [Rdr::download@74] Rdr::download SKIP for sequence as OPTIX_NON_INTEROP
    2019-06-01 22:49:36.720 INFO  [377734] [OpticksViz::downloadEvent@383] OpticksViz::downloadEvent (1) DONE 
    2019-06-01 22:49:36.720 INFO  [377734] [OpEngine::downloadEvent@149] .
    2019-06-01 22:49:36.720 INFO  [377734] [OContext::download@693] OContext::download PROCEED for sequence as OPTIX_NON_INTEROP
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value (Details: Function "RTresult _rtBufferGetDevicePointer(RTbuffer, int, void**)" caught exception: Cannot get device pointers from non-CUDA interop buffers.)
    
    Program received signal SIGABRT, Aborted.
    0x00007fffe2033207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libX11-devel-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libdrm-2.4.91-3.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2033207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20348f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe29427d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffe2940746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffe2940773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffe2940993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007ffff652e550 in optix::APIObj::checkError (this=0x3ad3ac0, code=RT_ERROR_INVALID_VALUE) at /home/blyth/local/opticks/externals/OptiX/include/optixu/optixpp_namespace.h:2151
    #7  0x00007ffff6570599 in OBufBase::getDevicePtr() () from /home/blyth/local/opticks/lib/../lib64/libOptiXRap.so
    #8  0x00007ffff657076e in OBufBase::bufspec() () from /home/blyth/local/opticks/lib/../lib64/libOptiXRap.so
    #9  0x00007ffff6552802 in OEvent::downloadHits (this=0x3c63ba0, evt=0x3ad3000) at /home/blyth/opticks/optixrap/OEvent.cc:412
    #10 0x00007ffff655239a in OEvent::download (this=0x3c63ba0) at /home/blyth/opticks/optixrap/OEvent.cc:354
    #11 0x00007ffff68a51a4 in OpEngine::downloadEvent (this=0x2d2b590) at /home/blyth/opticks/okop/OpEngine.cc:151
    #12 0x00007ffff79ccc5c in OKPropagator::downloadEvent (this=0x2d2e590) at /home/blyth/opticks/ok/OKPropagator.cc:108
    #13 0x00007ffff79cca64 in OKPropagator::propagate (this=0x2d2e590) at /home/blyth/opticks/ok/OKPropagator.cc:82
    #14 0x00007ffff7bd5829 in OKG4Mgr::propagate_ (this=0x7fffffffcbe0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:190
    #15 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcbe0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #16 0x00000000004039a7 in main (argc=35, argv=0x7fffffffcf18) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 


::

     tboolean-;tboolean-box --okg4 --compute
     tboolean-;tboolean-box --okg4 --load


Issue 6 : tboolean-box-p looking in OPTICKS_KEY geocache : FIXED with resource generalizations
-------------------------------------------------------------------------------------------------

::

    [blyth@localhost cfg4]$ tboolean-box-p
    args: /home/blyth/opticks/ana/tboolean.py --det tboolean-box --tag 1
    [2019-06-01 23:21:47,570] p434209 {/home/blyth/opticks/ana/base.py:340} INFO -  ( opticks_environment
    [2019-06-01 23:21:47,571] p434209 {/home/blyth/opticks/ana/base.py:345} INFO -  ) opticks_environment
    [2019-06-01 23:21:47,571] p434209 {/home/blyth/opticks/ana/tboolean.py:62} INFO - tag 1 src torch det tboolean-box c2max 2.0 ipython False 
    [2019-06-01 23:21:47,571] p434209 {/home/blyth/opticks/ana/ab.py:110} INFO - ab START
    [2019-06-01 23:21:47,572] p434209 {/home/blyth/opticks/ana/base.py:638} WARNING - failed to load json from /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/tboolean-box/torch/1/parameters.json
    Traceback (most recent call last):
      File "/home/blyth/opticks/ana/tboolean.py", line 64, in <module>
        ab = AB(ok)
      File "/home/blyth/opticks/ana/ab.py", line 116, in __init__
        self.load()
      File "/home/blyth/opticks/ana/ab.py", line 138, in load
        a = Evt(tag=atag, src=args.src, det=args.det, args=args, nom="A", smry=args.smry)
      File "/home/blyth/opticks/ana/evt.py", line 212, in __init__
        ok = self.init_metadata()
      File "/home/blyth/opticks/ana/evt.py", line 301, in init_metadata
        metadata = Metadata(self.tagdir)
      File "/home/blyth/opticks/ana/metadata.py", line 91, in __init__
        self.parameters = json_(os.path.join(self.path, "parameters.json"))
      File "/home/blyth/opticks/ana/base.py", line 639, in json_
        assert 0
    AssertionError
    [blyth@localhost cfg4]$ 


Hmm where should things (geometry, events, metadata) be saved for test running.
As test geometry has a basis geometry, first impulse is to put it within the corresponding 
geocache.  But in a multiuser environment will have a split in responsibilities, a small
number of administrators will create the basis geocaches and users who will not have 
write permission to the geocache will want to run tests using the materials from it. 

This makes me plump for CWD eg "tboolean-box--" directory named in the CSG csgpath argument 
after the FUNCNAME and events within it.




Issue 7 : resource generalizations to work with relative event paths
-----------------------------------------------------------------------------

* :doc:`strace-monitor-file-opens`



Issue 8 : tboolean-box-ip polarizations are totally off : FIXED 
-------------------------------------------------------------------

* :doc:`tboolean-box-ip-polarization-mismatch`










OBndLib_convert_issue_for_test_geometry_with_no_surfaces
===========================================================

::

    simon:ggeo blyth$ tboolean-;tboolean-media--

    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  

    args = opticks_main(csgpath="/tmp/blyth/opticks/tboolean-media--")

    omat = "Rock"
    osur = ""
    #isur = "perfectAbsorbSurface"
    isur = ""
    imat = "Pyrex"

    box = CSG("box", param=[0,0,0,400], boundary="/".join([omat,osur,isur,imat]), poly="MC", nx="20", emit=-1, emitconfig="photons=600000,wavelength=380,time=0.2,posdelta=0.1,sheetmask=0x1" )  
    CSG.Serialize( [box], args.csgpath )





::

    simon:ggeo blyth$ tboolean-;tboolean-media --okg4 -D

    ...

    (  0)           0           1           0           0 
    2017-11-10 16:09:30.004 ERROR [4187274] [*GBndLib::createBufferForTex2d@682] GBndLib::createBufferForTex2d NULL BUFFERS  mat 0x10d0118c0 sur 0x0
    Assertion failed: (orig && "OBndLib::convert orig buffer NULL"), function convert, file /Users/blyth/opticks/optixrap/OBndLib.cc, line 86.
    Process 25213 stopped
    * thread #1: tid = 0x3fe48a, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8cc60866:  jae    0x7fff8cc60870            ; __pthread_kill + 20
       0x7fff8cc60868:  movq   %rax, %rdi
       0x7fff8cc6086b:  jmp    0x7fff8cc5d175            ; cerror_nocancel
       0x7fff8cc60870:  retq   
    (lldb) bt
    * thread #1: tid = 0x3fe48a, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x000000010361a7ff libOptiXRap.dylib`OBndLib::convert(this=0x000000012001e190) + 367 at OBndLib.cc:86
        frame #5: 0x0000000103621a11 libOptiXRap.dylib`OScene::init(this=0x000000010966f270) + 7361 at OScene.cc:171
        frame #6: 0x000000010361fced libOptiXRap.dylib`OScene::OScene(this=0x000000010966f270, hub=0x000000010970ed20) + 317 at OScene.cc:85
        frame #7: 0x0000000103621d3d libOptiXRap.dylib`OScene::OScene(this=0x000000010966f270, hub=0x000000010970ed20) + 29 at OScene.cc:87
        frame #8: 0x0000000103bbad56 libOKOP.dylib`OpEngine::OpEngine(this=0x0000000109671450, hub=0x000000010970ed20) + 182 at OpEngine.cc:43
        frame #9: 0x0000000103bbb21d libOKOP.dylib`OpEngine::OpEngine(this=0x0000000109671450, hub=0x000000010970ed20) + 29 at OpEngine.cc:55
        frame #10: 0x0000000103f09a44 libOK.dylib`OKPropagator::OKPropagator(this=0x00000001096713f0, hub=0x000000010970ed20, idx=0x000000010d013a30, viz=0x00000001122980c0) + 196 at OKPropagator.cc:44
        frame #11: 0x0000000103f09bbd libOK.dylib`OKPropagator::OKPropagator(this=0x00000001096713f0, hub=0x000000010970ed20, idx=0x000000010d013a30, viz=0x00000001122980c0) + 45 at OKPropagator.cc:52
        frame #12: 0x00000001044d8ddf libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe460, argc=27, argv=0x00007fff5fbfe548) + 831 at OKG4Mgr.cc:37
        frame #13: 0x00000001044d8f53 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe460, argc=27, argv=0x00007fff5fbfe548) + 35 at OKG4Mgr.cc:41
        frame #14: 0x00000001000132ee OKG4Test`main(argc=27, argv=0x00007fff5fbfe548) + 1486 at OKG4Test.cc:56
        frame #15: 0x00007fff880d35fd libdyld.dylib`start + 1
        frame #16: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 5
    frame #5: 0x0000000103621a11 libOptiXRap.dylib`OScene::init(this=0x000000010966f270) + 7361 at OScene.cc:171
       168  
       169      LOG(debug) << "OScene::init (OBndLib)" ;
       170      m_olib = new OBndLib(context,m_ggeo->getBndLib());
       ///
       ///                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^WRONG     
       ///        Ahha : bndlib should come hub, not ggeo : THERE ARE PROBABLY MORE OF THESE
       ///
    -> 171      m_olib->convert();
       172      // this creates the BndLib dynamic buffers, which needs to be after OGeo
       173      // as that may add boundaries when using analytic geometry
       174  
    (lldb) 

    (lldb) f 4
    frame #4: 0x000000010361a7ff libOptiXRap.dylib`OBndLib::convert(this=0x000000012001e190) + 367 at OBndLib.cc:86
       83   
       84       NPY<float>* orig = m_lib->getBuffer() ;  // (123, 4, 2, 39, 4)
       85   
    -> 86       assert(orig && "OBndLib::convert orig buffer NULL");
       87   
       88       NPY<float>* buf = m_debug_buffer ? m_debug_buffer : orig ; 
       89   
    (lldb) 


Ahha : bndlib should come hub, not ggeo : THERE ARE PROBABLY MORE OF THESE 


False alarm::


    2017-11-10 16:09:29.717 INFO  [4187274] [OScene::init@111] OScene::init (OContext) stack_size_bytes: 2180
    2017-11-10 16:09:29.717 INFO  [4187274] [OScene::init@129] OScene::init ggeobase identifier : GGeoTest
    2017-11-10 16:09:29.717 INFO  [4187274] [OGeo::convert@172] OGeo::convert START  numMergedMesh: 1




Remove direct access to m_ggeo from the higher levels : everyting via hub 
---------------------------------------------------------------------------

OpticksIdx needs GGeo for NSensorList

::

    112 void OpticksIdx::indexEvtOld()
    113 {
    114     OpticksEvent* evt = getEvent();
    115     if(!evt) return ;
    116 
    117     // TODO: wean this off use of Types, for the new way (GFlags..)
    118     Types* types = m_ok->getTypes();
    119     Typ* typ = m_ok->getTyp();
    120 
    121     NPY<float>* ox = evt->getPhotonData();
    122 
    123     if(ox && ox->hasData())
    124     {
    125         PhotonsNPY* pho = new PhotonsNPY(ox);   // a detailed photon/record dumper : looks good for photon level debug 
    126         pho->setTypes(types);
    127         pho->setTyp(typ);
    128         evt->setPhotonsNPY(pho);
    129 
    130         GGeo* ggeo = m_hub->getGGeo();
    131 
    132         if(!ggeo) LOG(fatal) << "OpticksIdx::indexEvtOld"
    133                              << " MUST OpticksHub::loadGeometry before OpticksIdx::indexEvtOld "
    134                              ;
    135 
    136         assert(ggeo);
    137         HitsNPY* hit = new HitsNPY(ox, ggeo->getSensorList());
    138         evt->setHitsNPY(hit);
    139     }



Move sensorlist to hub ?

::

    simon:ggeo blyth$ opticks-find getSensorList 
    ./assimprap/AssimpGGeo.cc:    NSensorList* sens = gg->getSensorList();  
    ./assimprap/AssimpGGeo.cc:    NSensorList* sens = gg->getSensorList();  
    ./ggeo/GGeo.cc:NSensorList* GGeo::getSensorList()
    ./ggeo/GPmt.cc:   897     NSensorList* sens = gg->getSensorList();
      

    ./opticksgeo/OpticksIdx.cc:        HitsNPY* hit = new HitsNPY(ox, ggeo->getSensorList());


    ./ggeo/GGeo.hh:        NSensorList*  getSensorList();
    ./ggeo/GScene.cc:    m_sensor_list(ggeo->getSensorList()),

    simon:opticks blyth$ 





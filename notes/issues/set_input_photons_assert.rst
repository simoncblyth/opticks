set_input_photons_assert
===========================


::

    2021-06-07 16:53:08.070 INFO  [133081] [junoSD_PMT_v2_Opticks::Initialize@83]  tool 0x1e48ce0 input_photons 0x2fe6240 g4ok 0x4cdec30
    2021-06-07 16:53:08.070 INFO  [133081] [G4Opticks::setInputPhotons@1934]  input_photons 8,4,4

    Program received signal SIGSEGV, Segmentation fault.
    (gdb) bt
    #0  0x00007fffedd83ff8 in std::vector<int, std::allocator<int> >::size() const () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libEDMUtil.so
    #1  0x00007fffd6067962 in NPYBase::getNumItems (this=0x0, ifr=0, ito=1) at /home/blyth/opticks/npy/NPYBase.cpp:538
    #2  0x00007fffd60d9fad in NPY<unsigned long long>::expand (this=0x0, extra_items=0) at /home/blyth/opticks/npy/NPY.cpp:492
    #3  0x00007fffcdd920dc in CWriter::expand (this=0x14bd497f0, gs_photons=0) at /home/blyth/opticks/cfg4/CWriter.cc:117
                                                               ^^^^^^^^^^^^^^^^
    #4  0x00007fffcdd92182 in CWriter::BeginOfGenstep (this=0x14bd497f0) at /home/blyth/opticks/cfg4/CWriter.cc:136
    #5  0x00007fffcdd873a3 in CRecorder::BeginOfGenstep (this=0x14bd49680) at /home/blyth/opticks/cfg4/CRecorder.cc:169
    #6  0x00007fffcddb2aef in CManager::BeginOfGenstep (this=0x14bd493c0, genstep_index=0, gentype=84 'T', num_photons=0, offset=0) at /home/blyth/opticks/cfg4/CManager.cc:187
    #7  0x00007fffcddb6099 in CGenstepCollector::addGenstep (this=0x130f97000, numPhotons=0, gentype=84 'T') at /home/blyth/opticks/cfg4/CGenstepCollector.cc:302
                                                                               ^^^^^^^^^^^^^
    #8  0x00007fffcddb6b59 in CGenstepCollector::collectOpticksGenstep (this=0x130f97000, gs=0x2a6ee10) at /home/blyth/opticks/cfg4/CGenstepCollector.cc:548
    #9  0x00007fffce070061 in G4Opticks::setInputPhotons (this=0x4cdec30, input_photons=0x2fe6240) at /home/blyth/opticks/g4ok/G4Opticks.cc:1947
    #10 0x00007fffc214c00b in junoSD_PMT_v2_Opticks::Initialize (this=0x34b04f0) at ../src/junoSD_PMT_v2_Opticks.cc:91
    #11 0x00007fffc2146f0d in junoSD_PMT_v2::Initialize (this=0x34b0550, HCE=0x2a6e4d0) at ../src/junoSD_PMT_v2.cc:188
    #12 0x00007fffcda04a97 in G4SDStructure::Initialize(G4HCofThisEvent*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so
    #13 0x00007fffcda02f5b in G4SDManager::PrepareNewEvent() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so
    #14 0x00007fffd06ed85c in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #15 0x00007fffc269f760 in G4SvcRunManager::SimulateEvent(int) () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libG4Svc.so
    #16 0x00007fffc1bffa3c in DetSimAlg::execute (this=0x250f2a0) at ../src/DetSimAlg.cc:112
    #17 0x00007fffef13836d in Task::execute() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so
    #18 0x00007fffef13d568 in TaskWatchDog::run() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so
    #19 0x00007fffef137f49 in Task::run() () from /home/blyth/junotop/sniper/InstallArea/Linux-x86_64/lib/libSniperKernel.so


* trying to expand the event buffer prior to it existing perhaps ?
* input photons result in a very early BeginOfGenstep call 

  * need to defer the call until after the OpticksEvent is created
 

Failed to duplicate the issues in a fast cycle environment with G4OKTest::

    epsilon:src blyth$ opticks-
    epsilon:src blyth$ opticks-f setInputPhotons

    ./opticksgeo/OpticksGen.cc:    setInputPhotons(iox);
    ./opticksgeo/OpticksGen.cc:void OpticksGen::setInputPhotons(NPY<float>* ox)
    ./opticksgeo/OpticksGen.hh:        void                 setInputPhotons(NPY<float>* iox);

    ## this was the older approach  

    ./g4ok/G4Opticks.cc:void G4Opticks::setInputPhotons(const char* dir, const char* name)
    ./g4ok/G4Opticks.cc:    setInputPhotons(input_photons); 
    ./g4ok/G4Opticks.cc:void G4Opticks::setInputPhotons(const char* path)
    ./g4ok/G4Opticks.cc:    setInputPhotons(input_photons); 
    ./g4ok/G4Opticks.cc:G4Opticks::setInputPhotons
    ./g4ok/G4Opticks.cc:void G4Opticks::setInputPhotons(NPY<float>* input_photons)
    ./g4ok/G4Opticks.hh:        void setInputPhotons(const char* dir, const char* name) ;
    ./g4ok/G4Opticks.hh:        void setInputPhotons(const char* path) ;
    ./g4ok/G4Opticks.hh:        void setInputPhotons(NPY<float>* input_photons) ;

    ./g4ok/tests/G4OKTest.cc:    m_g4ok->setInputPhotons(path); 

    ## this test did not fail 

    ./optickscore/OpticksGenstep.cc:Invoked from G4Opticks::setInputPhotons 
    epsilon:opticks blyth$ 





::

    1932 void G4Opticks::setInputPhotons(NPY<float>* input_photons)
    1933 {
    1934     LOG(info)
    1935         << " input_photons " << ( input_photons ? input_photons->getShapeString() : "-" )
    1936         ;
    1937 
    1938     if( input_photons == nullptr )
    1939     {
    1940         LOG(error) << " null input_photons, ignore " ;
    1941         return ;
    1942     }
    1943 
    1944     unsigned tagoffset = 0 ;
    1945     const OpticksGenstep* gs = OpticksGenstep::MakeInputPhotonCarrier(input_photons, tagoffset );
    1946     assert( m_genstep_collector );
    1947     m_genstep_collector->collectOpticksGenstep(gs);
    1948 }


    397 OpticksGenstep* OpticksGenstep::MakeInputPhotonCarrier(NPY<float>* ip, unsigned tagoffset ) // static
    398 {
    399     unsigned num_photons = ip->getNumItems();
    400     LOG(LEVEL)
    401         << " num_photons " << num_photons
    402         << " input_photons " << ip->getShapeString()
    403         << " tagoffset " << tagoffset
    404         ;
    405 
    406     NStep onestep ;
    407     onestep.setGenstepType( OpticksGenstep_EMITSOURCE );
    408     onestep.setNumPhotons(  num_photons );
    409     onestep.fillArray();
    410     NPY<float>* gs = onestep.getArray();
    411 
    412 
    413     bool compute = true ;
    414     ip->setBufferSpec(OpticksEvent::SourceSpec(compute));
    415     ip->setArrayContentIndex( tagoffset );
    416 
    417     gs->setBufferSpec(OpticksEvent::GenstepSpec(compute));
    418     gs->setArrayContentIndex( tagoffset );
    419 
    420     OpticksActionControl oac(gs->getActionControlPtr());
    421     oac.add(OpticksActionControl::GS_EMITSOURCE_);       // needed ?
    422     LOG(LEVEL)
    423         << " gs " << gs
    424         << " oac.desc " << oac.desc("gs")
    425         << " oac.numSet " << oac.numSet()
    426         ;
    427 
    428     gs->setAux((void*)ip);  // under-radar association of input photons with the fabricated genstep 
    429 
    430     OpticksGenstep* ogs = new OpticksGenstep(gs);
    431     return ogs ;
    432 }



The automatic invokation of BeginOfGenstep from CGenstepCollector 
is convenient for C+S gensteps but its too early for input_photon 
torch gensteps.  
And there is already special casing to invoke BeginOfGenstep
for input photons in CManager::BeginOfEventAction::


    124 void CManager::BeginOfEventAction(const G4Event* event)
    125 {
    126     LOG(LEVEL) << " m_mode " << m_mode ;
    127     if(m_mode == 0 ) return ;
    128 
    129     m_ctx->setEvent(event);
    130 
    131     if(m_ok->isSave()) presave();   // creates the OpticksEvent
    132 
    133     if( m_ctx->_number_of_input_photons  > 0 )
    134     {
    135         LOG(LEVEL)
    136             << " mocking BeginOfGenstep as have input photon primaries "
    137             << CEvent::DescPrimary(event)
    138             ;
    139 
    140         unsigned genstep_index = 0 ;
    141         BeginOfGenstep(genstep_index, 'T', m_ctx->_number_of_input_photons, 0 );
    142     }
    143 }


::

    #0  0x00007fffedd83ff8 in std::vector<int, std::allocator<int> >::size() const () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/lib/libEDMUtil.so
    #1  0x00007fffd6067962 in NPYBase::getNumItems (this=0x0, ifr=0, ito=1) at /home/blyth/opticks/npy/NPYBase.cpp:538
    #2  0x00007fffd60d9fad in NPY<unsigned long long>::expand (this=0x0, extra_items=8) at /home/blyth/opticks/npy/NPY.cpp:492
    #3  0x00007fffcdd9214c in CWriter::expand (this=0x14bd497f0, gs_photons=8) at /home/blyth/opticks/cfg4/CWriter.cc:117
    #4  0x00007fffcdd921f2 in CWriter::BeginOfGenstep (this=0x14bd497f0) at /home/blyth/opticks/cfg4/CWriter.cc:136
    #5  0x00007fffcdd87413 in CRecorder::BeginOfGenstep (this=0x14bd49680) at /home/blyth/opticks/cfg4/CRecorder.cc:169
    #6  0x00007fffcddb2b5f in CManager::BeginOfGenstep (this=0x14bd493c0, genstep_index=0, gentype=84 'T', num_photons=8, offset=0) at /home/blyth/opticks/cfg4/CManager.cc:187
    #7  0x00007fffcddb6109 in CGenstepCollector::addGenstep (this=0x130f97000, numPhotons=8, gentype=84 'T') at /home/blyth/opticks/cfg4/CGenstepCollector.cc:302
    #8  0x00007fffcddb6fa4 in CGenstepCollector::collectTorchGenstep (this=0x130f97000, gs=0x2a6ee10) at /home/blyth/opticks/cfg4/CGenstepCollector.cc:583
    #9  0x00007fffce070061 in G4Opticks::setInputPhotons (this=0x4cdec30, input_photons=0x2fe6240) at /home/blyth/opticks/g4ok/G4Opticks.cc:1947
    #10 0x00007fffc214c00b in junoSD_PMT_v2_Opticks::Initialize (this=0x34b04f0) at ../src/junoSD_PMT_v2_Opticks.cc:91
    #11 0x00007fffc2146f0d in junoSD_PMT_v2::Initialize (this=0x34b0550, HCE=0x2a6e4d0) at ../src/junoSD_PMT_v2.cc:188
    #12 0x00007fffcda04a97 in G4SDStructure::Initialize(G4HCofThisEvent*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so
    #13 0x00007fffcda02f5b in G4SDManager::PrepareNewEvent() () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so
    #14 0x00007fffd06ed85c in G4EventManager::DoProcessing(G4Event*) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4event.so
    #15 0x00007fffc269f760 in G4SvcRunManager::SimulateEvent(int) () from /home/blyth/junotop/offline/InstallArea/Linux-x86_64/l



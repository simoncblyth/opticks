G4CXTest_raindrop_revive_as_demo_of_OPTICKS_INTEGRATION_MODE_3_for_validation
===============================================================================



FIXED : Issue 1 : assert from lack of genstep : due to omitted config
-----------------------------------------------------------------------

* maybe not updated for SEvt::beginOfEvent SEvt::endOfEvent rejig OR the multi-event index controls

::

    P[blyth@localhost ~]$ ~/o/g4cx/tests/G4CXTest_raindrop.sh dbg

    ...

    Local_DsG4Scintillation::Local_DsG4Scintillation level 0 verboseLevel 0
    2025-04-14 11:01:43.265 FATAL [331468] [U4Physics::CreateBoundaryProcess@412]  FAILED TO SPMTAccessor::Load from [$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/extra/jpmt] GEOM RaindropRockAirWater
    2025-04-14 11:01:43.265 INFO  [331468] [U4Physics::ConstructOp@310]  fBoundary 0x2ce26b0
    2025-04-14 11:01:43.265 INFO  [331468] [G4CXApp::G4CXApp@160]
    U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE:0
    $Name: geant4-10-04-patch-02 [MT]$ (25-May-2018)U4Recorder::Switches
    WITH_CUSTOM4
    NOT:WITH_PMTSIM
    NOT:PMTSIM_STANDALONE
    NOT:PRODUCTION
    NOT:WITH_INSTRUMENTED_DEBUG


    2025-04-14 11:01:43.265 INFO  [331468] [G4CXApp::BeamOn@343] [ OPTICKS_NUM_EVENT=1
    2025-04-14 11:01:43.654 INFO  [331468] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    G4CXTest: /home/blyth/opticks/sysrap/SEvent.cc:179: static NP* SEvent::MakeGenstep(int, int): Assertion `num_gs > 0' failed.

    Thread 1 "G4CXTest" received signal SIGABRT, Aborted.
    0x00007ffff23b9387 in raise () from /lib64/libc.so.6
    (gdb) bt
    #0  0x00007ffff23b9387 in raise () from /lib64/libc.so.6
    #1  0x00007ffff23baa78 in abort () from /lib64/libc.so.6
    #2  0x00007ffff23b21a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007ffff23b2252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff3a8e66c in SEvent::MakeGenstep (gentype=6, index_arg=0) at /home/blyth/opticks/sysrap/SEvent.cc:179
    #5  0x00007ffff3a8e30a in SEvent::MakeTorchGenstep (idx_arg=0) at /home/blyth/opticks/sysrap/SEvent.cc:143
    #6  0x000000000040a1e0 in G4CXApp::GeneratePrimaries (this=0x6c6400, event=0x3349260) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:236
    #7  0x00007ffff7059c4a in G4RunManager::GenerateEvent(int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #8  0x00007ffff705795c in G4RunManager::DoEventLoop(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #9  0x00007ffff70553ae in G4RunManager::BeamOn(int, char const*, int) () from /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno/lib64/libG4run.so
    #10 0x000000000040aa25 in G4CXApp::BeamOn (this=0x6c6400) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:344
    #11 0x000000000040ab31 in G4CXApp::Main () at /home/blyth/opticks/g4cx/tests/G4CXApp.h:351
    #12 0x000000000040acbf in main (argc=1, argv=0x7fffffff4488) at /home/blyth/opticks/g4cx/tests/G4CXTest.cc:13
    (gdb)

    (gdb) f 6
    #6  0x000000000040a1e0 in G4CXApp::GeneratePrimaries (this=0x6c6400, event=0x3349260) at /home/blyth/opticks/g4cx/tests/G4CXApp.h:236
    236         NP* gs = SEvent::MakeTorchGenstep(idx_arg) ;
    (gdb) p idx_arg
    $1 = 0
    (gdb) f 5
    #5  0x00007ffff3a8e30a in SEvent::MakeTorchGenstep (idx_arg=0) at /home/blyth/opticks/sysrap/SEvent.cc:143
    143 NP* SEvent::MakeTorchGenstep(   int idx_arg){ return MakeGenstep( OpticksGenstep_TORCH, idx_arg ) ; }
    (gdb) f 4
    #4  0x00007ffff3a8e66c in SEvent::MakeGenstep (gentype=6, index_arg=0) at /home/blyth/opticks/sysrap/SEvent.cc:179
    179     assert( num_gs > 0 );
    (gdb) p num_gs
    $2 = 0
    (gdb)


    (gdb) list
    174         << " num_ph/M " << num_ph/M
    175         << " num_gs " << num_gs
    176         << " dump " << dump
    177         ;
    178
    179     assert( num_gs > 0 );
    180
    181     NP* gs = NP::Make<float>(num_gs, 6, 4 );
    182     gs->set_meta<std::string>("creator", "SEvent::MakeGenstep" );
    183     gs->set_meta<int>("num_ph", num_ph );
    (gdb) list 160
    155
    156 **/
    157
    158
    159 NP* SEvent::MakeGenstep( int gentype, int index_arg )
    160 {
    161     bool with_index = index_arg != -1 ;
    162     if(with_index) assert( index_arg >= 0 );  // index_arg is 0-based
    163     int num_ph = with_index ? SEventConfig::NumPhoton(index_arg)  : ssys::getenvint("SEvent__MakeGenstep_num_ph", 100 ) ;
    164     int num_gs = with_index ? SEventConfig::NumGenstep(index_arg) : ssys::getenvint("SEvent__MakeGenstep_num_gs", 1   ) ;
    (gdb) p with_index
    $1 = true
    (gdb) p index_arg
    $2 = 0
    (gdb) p num_gs
    $4 = 0


Hmm torch running probably needs the comma delimited ph and gs lists
to have the same numbers of entries

Issue was that OPTICKS_NUM_GENSTEP was not defined, added comment::

     65 num=H1
     66 NUM=${NUM:-$num}
     67
     68 ## For torch running MUST NOW configure the below two NUM envvars
     69 ## with the same number of comma delimited values, or just 1 value
     70
     71 export OPTICKS_NUM_PHOTON=$NUM
     72 export OPTICKS_NUM_GENSTEP=1
     73 export OPTICKS_RUNNING_MODE="SRM_TORCH"
     74
     75 vars="$vars OPTICKS_NUM_PHOTON OPTICKS_NUM_GENSTEP OPTICKS_RUNNING_MODE"





FIXED : Issue 2 : VRAM OOM from debug arrays as forgot to set OPTICKS_MAX_SLOT which is needed when debugging
----------------------------------------------------------------------------------------------------------------

::

    P[blyth@localhost ~]$ ~/o/g4cx/tests/G4CXTest_raindrop.sh dbg
    ...

    2025-04-14 11:38:45.145 INFO  [410693] [G4CXApp::BeamOn@343] [ OPTICKS_NUM_EVENT=1
    2025-04-14 11:38:45.529 INFO  [410693] [G4CXApp::GeneratePrimaries@223] [ SEventConfig::RunningModeLabel SRM_TORCH eventID 0
    SGenerate::GeneratePhotons SGenerate__GeneratePhotons_RNG_PRECOOKED : NO
    U4VPrimaryGenerator::GeneratePrimaries_From_Photons ph (100000, 4, 4, )
     U4VPrimaryGenerator__GeneratePrimaries_From_Photons_DEBUG_GENIDX : -1 (when +ve, only generate tht photon idx)
    2025-04-14 11:38:45.601 INFO  [410693] [G4CXApp::GeneratePrimaries@253] ]  eventID 0
    2025-04-14 11:38:45.602 INFO  [410693] [U4Recorder::BeginOfEventAction_@333]  eventID 0
    2025-04-14 11:38:46.709 INFO  [410693] [U4Recorder::PreUserTrackingAction_Optical@450]  modulo 100000 : ulabel.id 0
    2025-04-14 11:38:47.017 INFO  [410693] [QSim::simulate@397] sslice {    0,    1,      0, 100000}
    2025-04-14 11:38:47.049 ERROR [410693] [QU::_cudaMalloc@272] save salloc record to /data/blyth/opticks/GEOM/RaindropRockAirWater/G4CXTest
    terminate called after throwing an instance of 'QUDA_Exception'
      what():  CUDA call (max_slot*max_record*sizeof(sphoton) ) failed with error: 'out of memory' (/home/blyth/opticks/qudarap/QU.cc:265)
    [salloc::desc alloc.size 13 label.size 13
    [salloc.meta
    evt.max_curand:1000000000
    evt.max_slot:197000000
    evt.max_photon:1000000000
    evt.num_photon:100000
    evt.max_curand/M:1000
    evt.max_slot/M:197
    evt.max_photon/M:1000
    evt.num_photon/M:0
    evt.max_record:10
    evt.max_rec:0
    evt.max_seq:1
    evt.max_prd:0
    evt.max_tag:0
    evt.max_flat:0
    evt.num_record:1000000
    evt.num_rec:0
    evt.num_seq:100000
    evt.num_prd:0
    evt.num_tag:0
    evt.num_flat:0
    ]salloc.meta

         [           size   num_items sizeof_item       spare]    size_GB    percent label
         [        (bytes)                                    ]   size/1e9

         [              8           1           8           0]       0.00       0.00 QBase::init/d_base
         [             24           1          24           0]       0.00       0.00 QRng::initMeta/d_qr
         [             64           1          64           0]       0.00       0.00 QTex::uploadMeta
         [             32           1          32           0]       0.00       0.00 QBnd::QBnd/d_qb
         [            432           1         432           0]       0.00       0.00 QDebug::QDebug/d_dbg
         [             24           1          24           0]       0.00       0.00 QCerenkov::QCerenkov/d_cerenkov.0
         [            256           1         256           0]       0.00       0.00 QEvent::QEvent/sevent
         [             64           1          64           0]       0.00       0.00 QSim::init.sim
         [        8294400     2073600           4           0]       0.01       0.01 Frame::DeviceAllo:num_pixels
         [      960000000    10000000          96           0]       0.96       0.67 QEvent::setGenstep/device_alloc_genstep_and_seed:quad6
         [     4000000000  1000000000           4           0]       4.00       2.78 QEvent::setGenstep/device_alloc_genstep_and_seed:int seed
         [    12608000000   197000000          64           0]      12.61       8.78 QEvent::device_alloc_photon/max_slot*sizeof(sphoton)
         [   126080000000  1970000000          64           0]     126.08      87.77 max_slot*max_record*sizeof(sphoton)

     tot     143656295304                                          143.66
    ]salloc::desc
     ;
    QU::_cudaMalloc_OOM_NOTES
    ==========================

    When running with debug arrays, such as the record array, enabled
    it is necessary to set max_slot to something reasonable, otherwise with the
    default max_slot of zero, it gets set to a high value (eg M197 with 24GB)
    appropriate for production running with the available VRAM.

    One million is typically reasonable for debugging::

       export OPTICKS_MAX_SLOT=M1




FIXED : Issue 3 : ana python was not loading the just created SEvt due to outdated AFOLD BFOLD
----------------------------------------------------------------------------------------------------

* observed from the sevt.py array ages logging
* fixed with OPTICKS_EVENT_NAME


FIXED : Issue 4 : not that SEvt chi2 comparison is using the old slow python not the C++ equivalent
------------------------------------------------------------------------------------------------------

Is this it ?::

    P[blyth@localhost tests]$ l sseq_index_test.*
    4 -rwxrwxr-x. 1 blyth blyth 2143 Apr 14 14:42 sseq_index_test.sh
    4 -rw-rw-r--. 1 blyth blyth 1622 Nov 27 15:19 sseq_index_test.cc
    4 -rw-rw-r--. 1 blyth blyth 1082 May 22  2024 sseq_index_test.py
    P[blyth@localhost tests]$

Added cf2 command using sseq_index_test.sh



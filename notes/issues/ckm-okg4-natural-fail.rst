ckm-okg4-natural-fail
========================

::

   ckm-okg4(){      OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save --natural ;}


Added natural to *ckm-okg4* to deal in the same gensteps as the others, but OKG4Test is 
not expecting "genstep" source running. Thats a recent capability.
Fully instrumented bi-executable running demands this to be made to work.::

    2019-03-13 19:31:17.636 INFO  [1512502] [OpticksEvent::setBufferControl@963]    genstep : (spec) : OPTIX_INPUT_ONLY UPLOAD_WITH_CUDA BUFFER_COPY_ON_DIRTY COMPUTE_MODE VERBOSE_MODE  : Evt /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1 20190313_193117 OKG4Test
    2019-03-13 19:31:17.636 INFO  [1512502] [OpticksRun::passBaton@170] OpticksRun::passBaton nopstep 0x1119b5020 genstep 0x111904990 source 0x0
    2019-03-13 19:31:17.636 INFO  [1512502] [OpticksEvent::setBufferControl@963]    genstep : (spec) : OPTIX_INPUT_ONLY UPLOAD_WITH_CUDA BUFFER_COPY_ON_DIRTY COMPUTE_MODE VERBOSE_MODE  : Evt /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/1 20190313_193117 OKG4Test
    2019-03-13 19:31:17.636 INFO  [1512502] [*CG4::propagate@304] Evt /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1 20190313_193117 OKG4Test  genstep 1,6,4 nopstep 0,4,4 photon 221,4,4 source NULL record 221,10,2,4 phosel 221,1,4 recsel 221,10,1,4 sequence 221,1,2 seed 221,1,1 hit 0,4,4
    2019-03-13 19:31:17.636 INFO  [1512502] [*CG4::propagate@322] CG4::propagate(0) /usr/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/792496b5e2cc08bdf5258cc12e63de9f/1/tmp/blyth/OKG4Test/evt/g4live/natural/-1
    2019-03-13 19:31:17.636 INFO  [1512502] [CGenerator::configureEvent@104] CGenerator:configureEvent fabricated TORCH genstep (STATIC RUNNING) 
    Assertion failed: (_gen == TORCH || _gen == G4GUN), function initEvent, file /Users/blyth/opticks/cfg4/CG4Ctx.cc, line 149.
    Process 25455 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff65123b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff65123b66 <+10>: jae    0x7fff65123b70            ; <+20>
        0x7fff65123b68 <+12>: movq   %rax, %rdi
        0x7fff65123b6b <+15>: jmp    0x7fff6511aae9            ; cerror_nocancel
        0x7fff65123b70 <+20>: retq   
    Target 0: (OKG4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff65123b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff652ee080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff6507f1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff650471ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106a63362 libCFG4.dylib`CG4Ctx::initEvent(this=0x0000000111904ee8, evt=0x00000001119941c0) at CG4Ctx.cc:149
        frame #5: 0x0000000106a682b8 libCFG4.dylib`CG4::initEvent(this=0x0000000111904ec0, evt=0x00000001119941c0) at CG4.cc:290
        frame #6: 0x0000000106a68ae7 libCFG4.dylib`CG4::propagate(this=0x0000000111904ec0) at CG4.cc:324
        frame #7: 0x00000001000e229a libOKG4.dylib`OKG4Mgr::propagate_(this=0x00007ffeefbfe7d8) at OKG4Mgr.cc:137
        frame #8: 0x00000001000e1ec6 libOKG4.dylib`OKG4Mgr::propagate(this=0x00007ffeefbfe7d8) at OKG4Mgr.cc:84
        frame #9: 0x0000000100014c7e OKG4Test`main(argc=6, argv=0x00007ffeefbfe8a8) at OKG4Test.cc:9
        frame #10: 0x00007fff64fd3015 libdyld.dylib`start + 1
        frame #11: 0x00007fff64fd3015 libdyld.dylib`start + 1
    (lldb) 

::

    139 void CG4Ctx::initEvent(const OpticksEvent* evt)
    140 {
    141     _ok_event_init = true ;
    142     _photons_per_g4event = evt->getNumPhotonsPerG4Event() ;
    143     _steps_per_photon = evt->getMaxRec() ;
    144     _record_max = evt->getNumPhotons();   // from the genstep summation
    145     _bounce_max = evt->getBounceMax();
    146 
    147     const char* typ = evt->getTyp();
    148     _gen = OpticksFlags::SourceCode(typ);
    149     assert( _gen == TORCH || _gen == G4GUN  );
    150 
    151     LOG(info) << "CG4Ctx::initEvent"
    152               << " _record_max (numPhotons from genstep summation) " << _record_max
    153               << " photons_per_g4event " << _photons_per_g4event
    154               << " steps_per_photon " << _steps_per_photon
    155               << " gen " << _gen
    156               ;
    157 }


::

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff70186b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff70351080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff700e21ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff700aa1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000106a6353d libCFG4.dylib`CG4Ctx::initEvent(this=0x0000000110f25ed8, evt=0x0000000111ad97a0) at CG4Ctx.cc:159
        frame #5: 0x0000000106a682a8 libCFG4.dylib`CG4::initEvent(this=0x0000000110f25eb0, evt=0x0000000111ad97a0) at CG4.cc:290
        frame #6: 0x0000000106a68ad7 libCFG4.dylib`CG4::propagate(this=0x0000000110f25eb0) at CG4.cc:324
        frame #7: 0x00000001000e229a libOKG4.dylib`OKG4Mgr::propagate_(this=0x00007ffeefbfe7e8) at OKG4Mgr.cc:137
        frame #8: 0x00000001000e1ec6 libOKG4.dylib`OKG4Mgr::propagate(this=0x00007ffeefbfe7e8) at OKG4Mgr.cc:84
        frame #9: 0x0000000100014c7e OKG4Test`main(argc=6, argv=0x00007ffeefbfe8b0) at OKG4Test.cc:9
        frame #10: 0x00007fff70036015 libdyld.dylib`start + 1
    (lldb) 


::

     33 OKG4Mgr::OKG4Mgr(int argc, char** argv)
     34     :
     35     m_log(new SLog("OKG4Mgr::OKG4Mgr")),
     36     m_ok(new Opticks(argc, argv)),
     37     m_run(m_ok->getRun()),
     38     m_hub(new OpticksHub(m_ok)),            // configure, loadGeometry and setupInputGensteps immediately
     39     m_load(m_ok->isLoad()),
     40     m_idx(new OpticksIdx(m_hub)),
     41     m_num_event(m_ok->getMultiEvent()),     // huh : m_gen should be in change of the number of events ? 
     42     m_gen(m_hub->getGen()),
     43     m_g4(m_load ? NULL : new CG4(m_hub)),   // configure and initialize immediately 
     44     m_generator( m_load ? NULL : m_g4->getGenerator()),
     45     m_viz(m_ok->isCompute() ? NULL : new OpticksViz(m_hub, m_idx, true)),    // true: load/create Bookmarks, setup shaders, upload geometry immediately 
     46     m_propagator(new OKPropagator(m_hub, m_idx, m_viz))
     47 {
     48     (*m_log)("DONE");
     49 }







::

    2019-03-14 14:49:53.634 INFO  [10744] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@168]  Pmin 1.512e-06 Pmax 2.0664e-05 wavelength_min(nm) 60 wavelength_max(nm) 820 preVelocity 276.074 postVelocity 273.253
    2019-03-14 14:49:53.634 ERROR [10744] [*CCerenkovGenerator::GetRINDEX@73]  aMaterial 0x110e7ba80 aMaterial.Name Water materialIndex 1 num_material 3 Rindex 0x110e7c3e0 Rindex2 0x110e7c3e0
    2019-03-14 14:49:53.634 FATAL [10744] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@210]  Pmin 1.512e-06 Pmin2 (MinLowEdgeEnergy) 2.034e-06 dif 5.21998e-07 epsilon 1e-06 
                                                   Pmin(nm) 820 Pmin2(nm) 609.558
    2019-03-14 14:49:53.634 FATAL [10744] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@220]  Pmax 2.0664e-05 Pmax2 (MaxLowEdgeEnergy) 4.136e-06 dif 1.6528e-05 epsilon 1e-06 
                                                   Pmax(nm) 60 Pmax2(nm) 299.768
    Assertion failed: (Pmax_match && "material mismatches genstep source material"), function GeneratePhotonsFromGenstep, file /Users/blyth/opticks/cfg4/CCerenkovGenerator.cc, line 233.
    Process 2898 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff52a49b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff52a49b66 <+10>: jae    0x7fff52a49b70            ; <+20>
        0x7fff52a49b68 <+12>: movq   %rax, %rdi
        0x7fff52a49b6b <+15>: jmp    0x7fff52a40ae9            ; cerror_nocancel
        0x7fff52a49b70 <+20>: retq   



Looks like an assert due to a range mismatch between an input material energy range and
a standardized one. ckm::

    095 G4MaterialPropertyVector* DetectorConstruction::MakeConstantProperty(float value)
     96 {
     97     using CLHEP::eV ;
     98 
     99     G4double photonEnergy[]   = { 2.034*eV , 4.136*eV };
    100     G4double propertyValue[] ={  value  , value    };


    In [1]: 1240./2.034
    Out[1]: 609.6361848574238

    In [2]: 1240./4.136
    Out[2]: 299.80657640232107




::

    107
    108 
    109     2019-03-14 14:49:52.371 INFO  [10744] [CWriter::initEvent@75] CWriter::initEvent dynamic STATIC(GPU style) _record_max 221 _bounce_max  9 _steps_per_photon 10 num_g4event 1
    110     2019-03-14 14:49:52.371 INFO  [10744] [CRec::initEvent@87] CRec::initEvent note recstp
    111     2019-03-14 14:49:52.372 INFO  [10744] [*CG4::propagate@330]  calling BeamOn numG4Evt 1
    112     2019-03-14 14:49:53.634 INFO  [10744] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    113     2019-03-14 14:49:53.634 ERROR [10744] [GBndLib::getMaterialIndexFromLine@715]  line 7 ibnd 1 numBnd 3 imatsur 3
    114     2019-03-14 14:49:53.634 INFO  [10744] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@135]  genstep_idx 0 num_gs 1 materialLine 7 materialIndex 1      post  0.000   0.000   0.000   115 
    116     2019-03-14 14:49:53.634 INFO  [10744] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@168]  
                               Pmin 1.512e-06 Pmax 2.0664e-05 wavelength_min(nm) 60 wavelength_max(nm) 820 preVelocity 2
    117     2019-03-14 14:49:53.634 ERROR [10744] [*CCerenkovGenerator::GetRINDEX@73]  aMaterial 0x110e7ba80 aMaterial.Name Water materialIndex 1 num_material 3 Rindex 0x110e7c3e0 Rindex2 0x110e118     2019-03-14 14:49:53.634 FATAL [10744] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@210]  Pmin 1.512e-06 Pmin2 (MinLowEdgeEnergy) 2.034e-06 dif 5.21998e-07 epsilon 1e-06 Pmin(nm) 
    119     2019-03-14 14:49:53.634 FATAL [10744] [*CCerenkovGenerator::GeneratePhotonsFromGenstep@220]  
                               Pmax 2.0664e-05 Pmax2 (MaxLowEdgeEnergy) 4.136e-06 dif 1.6528e-05 epsilon 1e-06 Pmax(nm) 120     

                      Assertion failed: (Pmax_match && "material mismatches genstep source material"), function GeneratePhotonsFromGenstep, 
                      file /Users/blyth/opticks/cfg4/CCerenkovGenerator.cc, line 233.

    121     Process 2898 stopped
    122     * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
    123         frame #0: 0x00007fff52a49b66 libsystem_kernel.dylib`__pthread_kill + 10
    124     libsystem_kernel.dylib`__pthread_kill:
    125     ->  0x7fff52a49b66 <+10>: jae    0x7fff52a49b70            ; <+20>
    126         0x7fff52a49b68 <+12>: movq   %rax, %rdi
    127         0x7fff52a49b6b <+15>: jmp    0x7fff52a40ae9            ; cerror_nocancel
    128         0x7fff52a49b70 <+20>: retq   
    129     Target 0: (OKG4Test) stopped.
    130     (lldb) bt
    131     * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
    132       * frame #0: 0x00007fff52a49b66 libsystem_kernel.dylib`__pthread_kill + 10
    133         frame #1: 0x00007fff52c14080 libsystem_pthread.dylib`pthread_kill + 333
    134         frame #2: 0x00007fff529a51ae libsystem_c.dylib`abort + 127
    135         frame #3: 0x00007fff5296d1ac libsystem_c.dylib`__assert_rtn + 320
    136         frame #4: 0x0000000106980e94 libCFG4.dylib`CCerenkovGenerator::GeneratePhotonsFromGenstep(gs=0x0000000111c3e610, idx=0) at CCerenkovGenerator.cc:233
    137         frame #5: 0x0000000106a6ca12 libCFG4.dylib`CGenstepSource::generatePhotonsFromOneGenstep(this=0x0000000111c3e950) at CGenstepSource.cc:94
    138         frame #6: 0x0000000106a6c90d libCFG4.dylib`CGenstepSource::GeneratePrimaryVertex(this=0x0000000111c3e950, event=0x0000000127535ed0) at CGenstepSource.cc:70
    139         frame #7: 0x0000000106a2b983 libCFG4.dylib`CPrimaryGeneratorAction::GeneratePrimaries(this=0x0000000111a21ad0, event=0x0000000127535ed0) at CPrimaryGeneratorAction.cc:15
    140         frame #8: 0x00000001086ffbd0 libG4run.dylib`G4RunManager::GenerateEvent(this=0x0000000110f371b0, i_event=0) at G4RunManager.cc:460
    141         frame #9: 0x00000001086fe9d6 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000110f371b0, i_event=0) at G4RunManager.cc:398
    142         frame #10: 0x00000001086fe825 libG4run.dylib`G4RunManager::DoEventLoop(this=0x0000000110f371b0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:367
    143         frame #11: 0x00000001086fcce1 libG4run.dylib`G4RunManager::BeamOn(this=0x0000000110f371b0, n_event=1, macroFile=0x0000000000000000, n_select=-1) at G4RunManager.cc:273
    144         frame #12: 0x0000000106a68c16 libCFG4.dylib`CG4::propagate(this=0x0000000110f36f90) at CG4.cc:331
    145         frame #13: 0x00000001000e229a libOKG4.dylib`OKG4Mgr::propagate_(this=0x00007ffeefbfe7e8) at OKG4Mgr.cc:137
    146         frame #14: 0x00000001000e1ec6 libOKG4.dylib`OKG4Mgr::propagate(this=0x00007ffeefbfe7e8) at OKG4Mgr.cc:84
    147         frame #15: 0x0000000100014c7e OKG4Test`main(argc=6, argv=0x00007ffeefbfe8b0) at OKG4Test.cc:9
    148         frame #16: 0x00007fff528f9015 libdyld.dylib`start + 1
    149     (lldb) 
    150 




Thoughts : March 19, 2019
--------------------------

Use of the standardized domain for the material properties is essential(?*) to being able 
to use GPU textures.  Actually it is not essential, just highly convenient as it means can 
put all material properties into a single GPU texture.  In principal could 
be less stringent, for example could demand that all properties of a single material use the same domain and then have
separate textures for each material. 

The situation:

* user defines some material properties in Geant4 way on some domain
* opticks interpolates onto the standard domain and wavelength raster 
* pre-standardized domain edges going into genstep ??
* material sanity check assert is tripped by edge comparison

Question:

* why/where is genstep recording a pre-standardized domain ?

From an alignment point of view where want Geant4 to be using precisely the
same material properties. This behooves that some standardization processing 
happens to Geant4 materials at initialization. Actually need 
(when in aligment mode) to effectively recreate the Geant4 materials 
from the Opticks standardized ones.  
Hmm: vaguely recall doing something like this previously : 
traversing and standardizing materials. Maybe that was in CFG4 approach ? 


March 20, 2019
-------------------

G4Opticks has *standardize_geant4_materials* switch::

     18 void RunAction::BeginOfRunAction(const G4Run*)
     19 {
     20 #ifdef WITH_OPTICKS
     21     LOG(info) << "." ;
     22     G4cout << "###[ RunAction::BeginOfRunAction G4Opticks.setGeometry" << G4endl ;       
     23     G4VPhysicalVolume* world = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
     24     assert( world ) ;
     25     bool standardize_geant4_materials = true ;   // required for alignment 
     26     G4Opticks::GetOpticks()->setGeometry(world, standardize_geant4_materials );
     27     G4cout << "###] RunAction::BeginOfRunAction G4Opticks.setGeometry" << G4endl ;      
     28 #endif
     29 }



 






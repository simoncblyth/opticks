CerenkovMinimal
==================
Objective
-----------

CerenkovMinimal aims to show minimal use of embedded Opticks
to act as an starting point for users familiar with Geant4 examples.

Overview
----------

The idea is to simplify Opticks usage by everything going 
through the G4Opticks interface. This intentionally constrains
Opticks functionality and flexibility.   


Full Opticks usage, via geocache
----------------------------------

The geocache and gensteps created from the simple CerenkovMinimal 
Geant4 geometry can be used from full unconstrained Opticks executables 
by setting the OPTICKS_KEY reported by CerenkovMinimal.


Review Sources
-----------------

CerenkovMinimal.cc
    main, includes only G4.hh

G4
    holds instances of all the below action classes, PhysicsList<L4Cerenkov>
    and the Ctx. Connects them up. 
    
PhysicsList
    mostly standard but with templated Cerenkov class 

L4Cerenkov
    minimally modified Cerenkov process that invokes
    G4Opticks::Get()->collectCerenkovStep

    Currently for validation the ordinary photon generation loop
    is performed, with G4Opticks::setAlignIndex(photon_record_id)
    being called at the head before any consumption of random numbers
    and G4Opticks::setAlignIndex(-1) being called at the tail of the loop  

Cerenkov
    not currently used. Looks like a shim subclass 
    of G4Cerenkov with PostStepGetPhysicalInteractionLength 
    reimplemented for easy prodding 

DetectorConstruction
    very standard, in the style of simple Geant4 examples

SensitiveDetector
    processHits invokes G4Opticks::collectHit

Ctx
    instance is resident of G4 and is passed as only argument to 
    all the action ctors. 

    Context struct used via setEvent setTrack setStep etc.. 
    to maintain convenient state access
 
    Ctx::setTrack currently kills non-optical tracks after the first 
    genstep has been collected : for debugging with a single genstep 

    Ctx::setTrackOptical invokes G4Opticks::setAlignIndex(photon_record_id)
    Ctx::postTrackOptical invokes G4Opticks::setAlignIndex(-1) 


TrackInfo
    used to pass the photon_record_id from the L4Cerenkov photon generation 
    loop to subsequent propagation.    

    G4VUserTrackInformation subclass that holds a photon_record_id, this 
    is convenient for example when multiple Geant4 events correspond 
    to a single Opticks event and also when doing reemission continuation : 
    ie maintaining the original identity of a photon through a reemission in 
    Geant4 to allow comparison with Opticks that does this naturally


RunAction
    BeginOfRunAction passes world volume with G4Opticks::setGeometry 
    EndOfRunAction invokes G4Opticks::Finalize

EventAction
    EndOfEventAction invokes G4Opticks::propagateOpticalPhotons

TrackingAction
    handoff G4Track to Ctx::setTrack and Ctx::postTrack

SteppingAction
    handoff G4Step to Ctx::setStep 

PrimaryGeneratorAction
    standard simple G4ParticleGun

Cerenkov.hh
    shim on top of G4Cerenkov for debug dumping only

L4Cerenkov.hh
    G4VProcess subclass based on G4Cerenkov that collects Cerenkov gensteps
    using G4Opticks. Currently the CPU photon generation loop is still being 
    done for comparison purposes.

PhysicsList.hh
    EM and Optical physics, ConstructOp is templated, currently using 
    L4Cerenkov 

SensitiveDetector.hh
    SensitiveDetector::ProcessHits is for standard Geant4 hits not ones from GPU.
    Fairly undeveloped, needs machinery to shovel GPU hits in bulk into 
    collections.




Aligned RNG stream
---------------------

Notice the calls to G4Opticks::setAlignIndex, which invoke CAlignEngine::SetSequenceIndex

* L4Cerenkov : with argument photon_record_id and -1 bracketing the body of the photon generation
* Ctx::setTrackOptical Ctx::postTrackOptical which get invoked while propagating

This is done to allow the RNG sequence for each photon to be continuous in order to 
make it possible to match with the RNG sequence used on GPU.  Note this approach 
has also been made to work across Scintillator reemission in CFG4.  



L4Cerenkov
--------------

* random alignment setup is done prior to any consumption of
  random numbers within the photon generation loop

::

    182 G4VParticleChange*
    183 L4Cerenkov::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    184 
    ...
    292         G4double MeanNumberOfPhotons1 =
    293                      GetAverageNumberOfPhotons(charge,beta1,aMaterial,Rindex);
    294         G4double MeanNumberOfPhotons2 =
    295                      GetAverageNumberOfPhotons(charge,beta2,aMaterial,Rindex);
    296 
    297 
    298 #ifdef WITH_OPTICKS
    299     unsigned opticks_photon_offset = 0 ;
    300     {
    301         const G4ParticleDefinition* definition = aParticle->GetDefinition();
    302         G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
    303         G4int materialIndex = aMaterial->GetIndex();
    304         G4cout << "L4Cerenkov::PostStepDoIt"
    305                << " dp (Pmax-Pmin) " << dp
    306                << G4endl
    307                ;
    308 
    309         opticks_photon_offset = G4Opticks::Get()->getNumPhotons();
    310         // total number of photons for all gensteps collected before this one
    311         // within this OpticksEvent (potentially crossing multiple G4Event) 
    312 
    313         G4Opticks::Get()->collectCerenkovStep(
    314                0,                  // 0     id:zero means use cerenkov step count 
    315                aTrack.GetTrackID(),
    316                materialIndex,
    317                NumPhotons,
    318 
    319                x0.x(),                // 1
    320                x0.y(),
    321                x0.z(),
    322                t0,
    323 
    324                deltaPosition.x(),     // 2
    325                deltaPosition.y(),
    326                deltaPosition.z(),
    327                aStep.GetStepLength(),
    328 
    329                definition->GetPDGEncoding(),   // 3
    330                definition->GetPDGCharge(),
    331                aTrack.GetWeight(),
    332                pPreStepPoint->GetVelocity(),
    333 
    334                BetaInverse,       // 4   
    335                Pmin,
    336                Pmax,
    337                maxCos,
    338 
    339                maxSin2,   // 5
    340                MeanNumberOfPhotons1,
    341                MeanNumberOfPhotons2,
    342                pPostStepPoint->GetVelocity()
    343         );
    344     }
    345 #endif
    346 
    347 
    348     // NB eventually the below CPU photon generation loop 
    349     //    will be skipped, it is kept for now to allow comparisons for validation
    350 
    351     for (G4int i = 0; i < NumPhotons; i++) {
    352 
    353         // Determine photon energy
    354 #ifdef WITH_OPTICKS
    355         unsigned record_id = opticks_photon_offset+i ;
    356         G4Opticks::Get()->setAlignIndex(record_id);
    357 #endif
    358 
    359         G4double rand;
    360         G4double sampledEnergy, sampledRI;
    361         G4double cosTheta, sin2Theta;
    362 
    363         // sample an energy
    364 
    365         do {
    366             rand = G4UniformRand();
    367             sampledEnergy = Pmin + rand * dp;
    368             sampledRI = Rindex->Value(sampledEnergy);
    369             cosTheta = BetaInverse / sampledRI;

    ...    standard Cerenkov generation .... 
    
    463         aParticleChange.AddSecondary(aSecondaryTrack);
    464 
    465 
    466 #ifdef WITH_OPTICKS
    467         aSecondaryTrack->SetUserInformation(new TrackInfo( record_id ) );
    468         G4Opticks::Get()->setAlignIndex(-1);
    469 #endif
    470 
    471 
    472     }  // CPU photon generation loop 
    473 
    474     if (verboseLevel>0) {
    475        G4cout <<"L4Cerenkov::PostStepDoIt DONE -- NumberOfSecondaries = "
    476               << aParticleChange.GetNumberOfSecondaries() << G4endl;
    477     }
    478 
    479 
    480 #ifdef WITH_OPTICKS
    481        G4cout
    482            << "L4Cerenkov::PostStepDoIt G4Opticks.collectSecondaryPhotons"
    483            << G4endl
    484            ;
    485 
    486         G4Opticks::Get()->collectSecondaryPhotons(pParticleChange) ;
    487 #endif
    488 
    489         return pParticleChange;
    490 }




Following Logging Output
----------------------------

The below is a selection of the logging output from CernkovMinimal
with some explanations.


::

    epsilon:~ blyth$ oe-      # setup the environment
    epsilon:~ blyth$ CerenkovMinimal

    **************************************************************
     Geant4 version Name: geant4-10-04-patch-02    (25-May-2018)
                           Copyright : Geant4 Collaboration

    ...
    2020-06-06 21:03:42.503 INFO  [61988576] [OpEngine::uploadEvent@136] .
    2020-06-06 21:03:42.512 INFO  [61988576] [OpEngine::propagate@145] [
    2020-06-06 21:03:42.512 INFO  [61988576] [OpSeeder::seedPhotonsFromGenstepsViaOptiX@173] SEEDING TO SEED BUF  
    2020-06-06 21:03:42.524 INFO  [61988576] [OpEngine::propagate@155] ( propagator.launch 
    2020-06-06 21:03:43.533 INFO  [61988576] [OPropagator::prelaunch@185] 1 : (0;221,1) 

         ## (221,1) are the (width,height) dimensions of the OptiX launch 
         ## the number of photons is always known ahead of time, coming from the 
         ## gensteps collected from Geant4

    2020-06-06 21:03:43.533 INFO  [61988576] [BTimes::dump@177] OPropagator::prelaunch
                  validate000                  2.4e-05
                   compile000                    2e-06
                 prelaunch000                 0.800931
    2020-06-06 21:03:43.533 INFO  [61988576] [OPropagator::launch@214] LAUNCH NOW -
    2020-06-06 21:03:43.535 INFO  [61988576] [OPropagator::launch@223] LAUNCH DONE
    2020-06-06 21:03:43.535 INFO  [61988576] [OPropagator::launch@225] 1 : (0;221,1) 
    2020-06-06 21:03:43.535 INFO  [61988576] [BTimes::dump@177] OPropagator::launch
                    launch001                 0.001449

         ## thats the time of the launch

    2020-06-06 21:03:43.535 INFO  [61988576] [OpEngine::propagate@157] ) propagator.launch 
    2020-06-06 21:03:43.535 INFO  [61988576] [OpIndexer::indexSequenceCompute@237] OpIndexer::indexSequenceCompute
    2020-06-06 21:03:43.539 INFO  [61988576] [OpEngine::propagate@160] ]
    2020-06-06 21:03:43.539 INFO  [61988576] [OpEngine::downloadEvent@186] .
    2020-06-06 21:03:43.540 INFO  [61988576] [OEvent::downloadHits@396]  nhit 44 --dbghit N hitmask 0x40 SD SURFACE_DETECT

         ## the hitmask determines what is regarded as a hit, this can be changed with the "--dbghitmask" option

    2020-06-06 21:03:43.540 FATAL [61988576] [OpPropagator::propagate@78] evtId(0) DONE nhit: 44
    2020-06-06 21:03:43.540 INFO  [61988576] [OpticksEvent::saveHitData@1689]  num_hit 0 ht 0,4,4 tag -1

          ## this num_hit 0 for sub-event tag -1 (4) is expected see notes below
          ## This is because the Opticks "G4" event (with negated tag) m_g4evt in G4Opticks is not filled 
          ## that is only currently done with CFG4 fully instrumented running. 

    2020-06-06 21:03:43.542 ERROR [61988576] [OpticksEvent::saveIndex@2382] SKIP as not indexed 
    2020-06-06 21:03:43.543 INFO  [61988576] [OpticksEvent::makeReport@1773] tagdir /tmp/blyth/opticks/source/evt/g4live/natural/-1
    2020-06-06 21:03:43.554 INFO  [61988576] [OpticksEvent::saveHitData@1689]  num_hit 44 ht 44,4,4 tag 1

          ## num_hit 44 for sub-event tag 1 (Opticks)

    2020-06-06 21:03:43.558 INFO  [61988576] [OpticksEvent::makeReport@1773] tagdir /tmp/blyth/opticks/source/evt/g4live/natural/1
    2020-06-06 21:03:43.570 ERROR [61988576] [GGeo::anaEvent@2069]  evt 0x7f9ce7ddfb50
    2020-06-06 21:03:43.570 INFO  [61988576] [OpticksAna::run@92]  anakey (null) enabled N
    ...
    2020-06-06 21:03:43.577 INFO  [61988576] [Opticks::saveParameters@1185]  postpropagate save parameters.json into TagZeroDir /tmp/blyth/opticks/source/evt/g4live/natural/0
    2020-06-06 21:03:43.578 INFO  [61988576] [OpticksEvent::saveHitData@1689]  num_hit 148 ht 148,4,4 tag -1

          ## these 148 hits from standard G4 are "artificially" stuffed into m_g4evt in order to 


    EventAction::EndOfEventAction num_hits 44 hits 0x7f9cec9a47a0

    ###] EventAction::EndOfEventAction G4Opticks.propagateOpticalPhotons

    EventAction::EndOfEventAction DumpHitCollections 
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionA hcid    0 hc 0x7f9ce990bcc0 hc.entries 118
    SensitiveDetector::DumpHitCollections query SD0/OpHitCollectionB hcid    1 hc 0x7f9ce990bd08 hc.entries 30


    ###[ RunAction::EndOfRunAction G4Opticks.Finalize



In the above logging there is a clear discrepancy with 44 hits from Opticks and 148 from Geant4.
There are many possible causes of this.   Debugging to attain a match is almost impossible 
when the only information available is the hits obtained.  It is far easier when the 
full information of the optical photon propagation is recorded. 
The CFG4 package does full step-by-step recording Opticks and Geant4 propagations
into two OpticksEvent instances.
Due to its complexity this is not yet used for G4Opticks based running. 

Other tricks very useful to attain a match used in CFG4 running are:

* performing random aligned simulations that consume the same random numbers 
* using common input photons generated on CPU

Random aligned simulations mean that comparisons are not clouded by 
statistics, allowing direct comparison of positions, times, wavelengths, polarizations.


What are hits ?
-------------------

The hitmask is used to select which photons are regarded as "hits" and then only
these are downloaded from GPU to CPU.

optixrap/OEvent.cc::

     65 OEvent::OEvent(Opticks* ok, OContext* ocontext)
     66    :
     67    m_log(new SLog("OEvent::OEvent", "", LEVEL)),
     68    m_ok(ok),
     69    //m_hitmask(SURFACE_DETECT),
     70    //m_hitmask(TORCH | BULK_SCATTER | BOUNDARY_TRANSMIT | SURFACE_ABSORB),
     71    m_hitmask(ok->getDbgHitMask()),
     72    m_compute(ok->isCompute()),
     73    m_dbghit(m_ok->isDbgHit()),            // --dbghit

The default "ok->getDbgHitMask()" is "SD" for SURFACE_DETECT which can be seen from, optickscore/OpticksCfg.cc::

      87        m_mask(""),
      88        m_dbghitmask("SD"),
      89        //m_dbghitmask("TO,SC,BT,SA"),  // see OEvent::OEvent
      90        m_x4polyskip(""),
      91        m_csgskiplv(""),

The downloading is done in optixrap/OEvent.cc::

    473 /**
    474 OEvent::downloadHits
    475 -------------------------
    476
    477 Downloading hits is special because a selection of the
    478 photon buffer is required, necessitating
    479 the use of Thrust stream compaction. This avoids allocating
    480 memory for all the photons on the host, just need to allocate
    481 for the hits.
    482
    483 In interop need CUDA/Thrust access to underlying OpenGL buffer.
    484 In compute need CUDA/Thrust access to the OptiX buffer.
    485
    486 **/
    487
    488 unsigned OEvent::downloadHitsCompute(OpticksEvent* evt)
    489 {
    490     OK_PROFILE("_OEvent::downloadHitsCompute");
    491
    492     NPY<float>* hit = evt->getHitData();
    493
    494     CBufSpec cpho = m_photon_buf->bufspec();
    495     assert( cpho.size % 4 == 0 );
    496     cpho.size /= 4 ;    //  decrease size by factor of 4, increases cpho "item" from 1*float4 to 4*float4
    497
    498     bool verbose = m_dbghit ;
    499     TBuf tpho("tpho", cpho );
    500     unsigned nhit = tpho.downloadSelection4x4("OEvent::downloadHits", hit, m_hitmask, verbose);
    501     // hit buffer (0,4,4) resized to fit downloaded hits (nhit,4,4)
    502     assert(hit->hasShape(nhit,4,4));
    503
    504     OK_PROFILE("OEvent::downloadHitsCompute");
    505
    506     LOG(LEVEL)
    507          << " nhit " << nhit
    508          << " hit " << hit->getShapeString()
    509          ;
    510
    511     return nhit ;
    512 }

The resulting (nhit,4,4) hits buffer contains all the items of the photon buffer matching the hitmask.
The photon and hits buffers contain photon structs 

optixrap/cu/photon.h::

     41 struct Photon
     42 {
     43    float3 position ;
     44    float  time ;
     45 
     46    float3 direction ;
     47    float  weight ;
     48
     49    float3 polarization ;
     50    float  wavelength ;
     51
     52    quad flags ;     // x:boundary  y:photon_id   z:m1   w:history
     53                     //             [debug-only]
     54 };
     55
     56


Controlling the embedded Opticks
-------------------------------------

G4Opticks uses an embedded Opticks instance which has its own commandline.
Some of the first logging output from G4Opticks details the commandline that is in use.
For example::

    2021-03-16 15:02:01.049 INFO  [33575005] [G4Opticks::EmbeddedCommandLine@130] Using ecl :[ --compute --embedded --xanalytic --production --nosave] OPTICKS_EMBEDDED_COMMANDLINE
    2021-03-16 15:02:01.049 INFO  [33575005] [G4Opticks::EmbeddedCommandLine@131]  mode(Pro/Dev/Asis) P using "pro" (production) commandline without event saving 
    2021-03-16 15:02:01.049 INFO  [33575005] [G4Opticks::EmbeddedCommandLine@136] Using extra1 argument :[--skipaheadstep 1000]
    2021-03-16 15:02:01.049 INFO  [33575005] [G4Opticks::EmbeddedCommandLine@146] Using eclx envvar :[] OPTICKS_EMBEDDED_COMMANDLINE_EXTRA
    2021-03-16 15:02:01.049 INFO  [33575005] [*G4Opticks::InitOpticks@231] instanciate Opticks using embedded commandline only 
     --compute --embedded --xanalytic --production --nosave  --skipaheadstep 1000 
     

Two envvars can be used to control this commandline.

OPTICKS_EMBEDDED_COMMANDLINE
    Two special values "pro" and "dev" correspond to standard commandlines that are
    expected to be appropriate for production and development.  Note that the "dev" mode
    has saving enabled and will save large numbers of event files. The "pro" mode should 
    save very few files.
 
    The default "pro" mode corresponds to actual commandline "--compute --embedded --xanalytic --production --nosave"
    The "dev" mode corresponds to " --compute --embedded --xanalytic --save --natural --printenabled --pindex 0"

OPTICKS_EMBEDDED_COMMANDLINE_EXTRA
    This envvar (for expert use only) can be used to add some additional options to the commandline. 
    Duplicated commandline options will trip asserts.   
  
    
Typically when you wish to enable full event saving for debugging and analysis simply::

    export OPTICKS_EMBEDDED_COMMANDLINE=dev 
    CerenkovMinimal 



Examining Full Event Data
-----------------------------

When saving is enabled around 20 NumPy arrays are written for every event.::

    epsilon:~ blyth$ ls -l  /tmp/blyth/opticks/source/evt/g4live/natural/1/*.npy
    -rw-r--r--  1 blyth  wheel    144 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileLisLabels.npy
    -rw-r--r--  1 blyth  wheel     88 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileLis.npy
    -rw-r--r--  1 blyth  wheel    144 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileAccLabels.npy
    -rw-r--r--  1 blyth  wheel     96 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileAcc.npy
    -rw-r--r--  1 blyth  wheel     80 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileLabels.npy
    -rw-r--r--  1 blyth  wheel     80 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfile.npy
    -rw-r--r--  1 blyth  wheel     96 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/idom.npy
    -rw-r--r--  1 blyth  wheel    128 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/fdom.npy
    -rw-r--r--  1 blyth  wheel   2680 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/rs.npy
    -rw-r--r--  1 blyth  wheel    340 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/ps.npy
    -rw-r--r--  1 blyth  wheel     80 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/wy.npy
    -rw-r--r--  1 blyth  wheel     80 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/dg.npy
    -rw-r--r--  1 blyth  wheel   1120 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/ph.npy
    -rw-r--r--  1 blyth  wheel  10480 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/rx.npy
    -rw-r--r--  1 blyth  wheel   4240 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/ox.npy
    -rw-r--r--  1 blyth  wheel    368 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/gs.npy
    -rw-r--r--  1 blyth  wheel     80 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/hy.npy
    -rw-r--r--  1 blyth  wheel    528 Mar 16 15:15 /tmp/blyth/opticks/source/evt/g4live/natural/1/ht.npy
    epsilon:~ blyth$ 

The np.py script can be used to dump the shapes of the arrays::

    epsilon:~ blyth$ np.py /tmp/blyth/opticks/source/evt/g4live/natural/1/*.npy
    a :        /tmp/blyth/opticks/source/evt/g4live/natural/1/ox.npy :           (65, 4, 4) : 664a09f7c9df20fd7d7dd2b0f81ba924 : 20210316-1515 
    b :        /tmp/blyth/opticks/source/evt/g4live/natural/1/ph.npy :           (65, 1, 2) : 34e62cf02f2704c8dfd4416af0669462 : 20210316-1515 
    c :        /tmp/blyth/opticks/source/evt/g4live/natural/1/ps.npy :           (65, 1, 4) : 9001be6559eaf49850fe8f2f3f327001 : 20210316-1515 
    d :        /tmp/blyth/opticks/source/evt/g4live/natural/1/rs.npy :       (65, 10, 1, 4) : b97c5cc9f518cfcb77dc0e4d8194fecc : 20210316-1515 
    e :        /tmp/blyth/opticks/source/evt/g4live/natural/1/rx.npy :       (65, 10, 2, 4) : a4878228712f585f46567e59bde95655 : 20210316-1515 
    f :        /tmp/blyth/opticks/source/evt/g4live/natural/1/ht.npy :            (7, 4, 4) : be3c6de10d2e706f03a416508cfca316 : 20210316-1515 
    g :      /tmp/blyth/opticks/source/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : 098ac4cd701409d36788edcea0d51a31 : 20210316-1515 
    h :        /tmp/blyth/opticks/source/evt/g4live/natural/1/gs.npy :            (3, 6, 4) : eb7c1b49d19a09e5886b6468a398269f : 20210316-1515 


These are persisted OpticksEvent. See optickscore/OpticksEvent.cc to follow the details of what everything is.
Opticks has many python analysis scripts that use NumPy to interpret these arrays. However initially 
it is useful to use a manual approach to examine the arrays.

Loading photon arrays for first three events::

    In [1]: import numpy as np ; from glob import glob
    In [2]: ox_paths = sorted(glob("/tmp/blyth/opticks/source/evt/g4live/natural/?/ox.npy"))

    In [3]: ox_paths
    Out[3]:
    ['/tmp/blyth/opticks/source/evt/g4live/natural/1/ox.npy',
     '/tmp/blyth/opticks/source/evt/g4live/natural/2/ox.npy',
     '/tmp/blyth/opticks/source/evt/g4live/natural/3/ox.npy']

    In [4]: a = np.load(ox_paths[0])

    In [5]: b = np.load(ox_paths[1])

    In [6]: v = np.load(ox_paths[2])

    In [7]: c = np.load(ox_paths[2])

    In [8]: a.shape
    Out[8]: (65, 4, 4)

    In [9]: b.shape
    Out[9]: (71, 4, 4)

    In [10]: c.shape
    Out[10]: (67, 4, 4)


Dumping the first photon from each event::

    In [11]: a[0]
    Out[11]:
    array([[127.661, -35.996,  90.   ,   0.728],
           [  0.796,  -0.225,   0.562,   1.   ],
           [ -0.571,   0.027,   0.82 ,  79.028],
           [    nan,   0.   ,   0.   ,   0.   ]], dtype=float32)

    In [12]: b[0]
    Out[12]:
    array([[136.877,  -9.657,  90.   ,   0.745],
           [  0.834,  -0.059,   0.549,   1.   ],
           [ -0.552,  -0.089,   0.829,  79.028],
           [    nan,   0.   ,   0.   ,   0.   ]], dtype=float32)

    In [13]: c[0]
    Out[13]:
    array([[131.949,  -9.657,  90.   ,   0.727],
           [  0.824,  -0.06 ,   0.563,   1.   ],
           [ -0.566,  -0.088,   0.819,  79.028],
           [    nan,   0.   ,   0.   ,   0.   ]], dtype=float32)


Notice that the last lines contains int32 or uint32 so when viewed as float 
will often appear as nan.  The array can be viewed as a different type to see the values of the flags:: 

    In [14]: a[0].view(np.int32)
    Out[14]:
    array([[ 1124029005, -1039139963,  1119092736,  1060782814],
           [ 1061935996, -1100614914,  1057998924,  1065353216],
           [-1089327308,  1021211713,  1062341336,  1117654571],
           [    -196607,           2,           0,         129]], dtype=int32)


    In [23]: np.set_printoptions(precision=8, suppress=True)     ## control number fomatting 
    In [24]: a[0]                                                                                                                                                                                           
    Out[24]: 
    array([[127.66074   , -35.995625  ,  90.        ,   0.7275828 ],
           [  0.7963178 ,  -0.22455975,   0.56165004,   1.        ],
           [ -0.57103276,   0.02715504,   0.82047796,  79.02767   ],
           [         nan,   0.        ,   0.        ,   0.        ]], dtype=float32)

    In [25]: a[1]                                                                                                                                                                                           
    Out[25]: 
    array([[303.0482    , 116.91953   ,  90.        ,   1.5299634 ],
           [  0.89899653,   0.34706625,   0.26711446,   1.        ],
           [ -0.43312532,   0.794904  ,   0.42488822,  64.74065   ],
           [         nan,   0.        ,   0.        ,   0.        ]], dtype=float32)



For plotting of the data use matplotlib for 2d histograms or pyvista for GPU 
accelerated plots of 3d data.



Examining Hit Data
---------------------

CerenkovMinimal persists OpticksEvent instances from both the Opticks and Geant4 simulations, 
the convention of using positive event tags for Opticks (1) and corresponding negated tags for Geant4 (-1) is used::

    epsilon:g4ok blyth$ l /tmp/blyth/opticks/source/evt/g4live/natural/1/*.npy
    -rw-r--r--  1 blyth  wheel    144 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileLisLabels.npy
    -rw-r--r--  1 blyth  wheel     88 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileLis.npy
    -rw-r--r--  1 blyth  wheel    144 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileAccLabels.npy
    -rw-r--r--  1 blyth  wheel     96 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileAcc.npy
    -rw-r--r--  1 blyth  wheel   5520 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfileLabels.npy
    -rw-r--r--  1 blyth  wheel   1440 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/OpticksProfile.npy
    -rw-r--r--  1 blyth  wheel     96 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/idom.npy
    -rw-r--r--  1 blyth  wheel    128 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/fdom.npy
    -rw-r--r--  1 blyth  wheel   8920 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/rs.npy
    -rw-r--r--  1 blyth  wheel    964 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/ps.npy
    -rw-r--r--  1 blyth  wheel   3616 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/ph.npy
    -rw-r--r--  1 blyth  wheel  35440 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/rx.npy
    -rw-r--r--  1 blyth  wheel  14224 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/ox.npy
    -rw-r--r--  1 blyth  wheel    176 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/gs.npy
    -rw-r--r--  1 blyth  wheel   2896 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/1/ht.npy
    epsilon:g4ok blyth$ 

    epsilon:g4ok blyth$ l /tmp/blyth/opticks/source/evt/g4live/natural/-1/*.npy
    -rw-r--r--  1 blyth  wheel  14224 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/so.npy
    -rw-r--r--  1 blyth  wheel   9552 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/ht.npy
    -rw-r--r--  1 blyth  wheel    144 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/OpticksProfileLisLabels.npy
    -rw-r--r--  1 blyth  wheel     88 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/OpticksProfileLis.npy
    -rw-r--r--  1 blyth  wheel    144 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/OpticksProfileAccLabels.npy
    -rw-r--r--  1 blyth  wheel     96 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/OpticksProfileAcc.npy
    -rw-r--r--  1 blyth  wheel   5392 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/OpticksProfileLabels.npy
    -rw-r--r--  1 blyth  wheel   1408 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/OpticksProfile.npy
    -rw-r--r--  1 blyth  wheel     96 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/idom.npy
    -rw-r--r--  1 blyth  wheel    128 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/fdom.npy
    -rw-r--r--  1 blyth  wheel     80 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/ph.npy
    -rw-r--r--  1 blyth  wheel     80 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/rx.npy
    -rw-r--r--  1 blyth  wheel     80 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/ox.npy
    -rw-r--r--  1 blyth  wheel    176 Jun  7 14:53 /tmp/blyth/opticks/source/evt/g4live/natural/-1/gs.npy
    epsilon:g4ok blyth$ 


The Geant4 OpticksEvent instance m_g4evt is mostly empty as full instrumentation is 
not used.  But the hit data in ht.npy is present.

As the NPY (NumPy) serialization is used for binary array data the persisted 
arrays are easily examined from ipython using NumPy

::

    epsilon:bin blyth$ ipython

    # import numpy as np      # done within the ipython profile setup  

    In [1]: ht = np.load("/tmp/blyth/opticks/source/evt/g4live/natural/1/ht.npy")

    In [2]: ht
    Out[2]: 
    array([[[ 303.0482,  116.9195,   90.    ,    1.53  ],  
            [   0.899 ,    0.3471,    0.2671,    1.    ],  
            [  -0.4331,    0.7949,    0.4249,   64.7406],
            [      nan,    0.    ,    0.    ,    0.    ]], 

           [[ 153.1065,   16.8344,   90.    ,    0.8098],
            [   0.858 ,    0.0946,    0.5049,    1.    ],  
            [  -0.4882,    0.4558,    0.7442,  117.8821],
            [      nan,    0.    ,    0.    ,    0.    ]], 

    ...

    In [3]: ht.shape
    Out[3]: (44, 4, 4)     ## (nhit, 4,4)


    In [2]: ht[:,0]    # row 0 : position, time
    Out[2]: 
    array([[ 303.0482,  116.9195,   90.    ,    1.53  ],
           [ 153.1065,   16.8344,   90.    ,    0.8098],
           [ 128.8066,  -28.0179,   90.    ,    0.7245],
           [ 132.6022,  -13.899 ,   90.    ,    0.7302],
           [ 248.9373,   86.2949,   90.    ,    1.2638],
           [ 210.6323,   62.5527,   90.    ,    1.0778],
           [ 129.7702,  -21.9672,   90.    ,    0.7237],
           ...
           [ 127.8584,  -37.3653,   90.    ,    0.7297],
           [ -29.2682,  152.4176,   90.    ,   13.2224],
           [ 135.6495,   -6.84  ,   90.    ,    0.7396],
           [ 128.1061,  -33.5526,   90.    ,    0.7268],
           [ 127.9607,  -43.2016,   90.    ,    0.7367],
           [ 328.9369, -337.5704,   90.    ,    2.1781],
           [ 131.8775,  -65.5113,   90.    ,    0.7834],
           [ 188.6135, -165.1583,   90.    ,    1.2091]], dtype=float32)

    In [3]: ht[:,0].shape
    Out[3]: (44, 4)


    ## The nan and wierd values of the flags in the last row when viewed as floats are expected.
    ## Viewing them as integers is needed, which is easily done with *view*

    In [4]: ht.view(np.int32)[:,3]
    Out[4]: 
    array([[      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,     1089],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,     1089],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65],
           [      -3,        0, 67305985,       65]], dtype=int32)

           ## boundary, sensor idx, debug,  history mask  


    In [12]: "%x" % 67305985
    Out[12]: '4030201'     # view in hex 



Meanings of the photon flags
------------------------------

p.flags.i.x 
    signed integer boundary index of the last intersection boundary 
    
p.flags.u.y
    sensor index, intended to hold something like a PMT identifier (TODO: revive this)

p.flags.u.z
    debugging bit field, currently 3 of the 4 bytes hold placeholder values

p.flags.u.w
    photon history mask constructed from bitwise OR of state flags for every step of the propagation 



p.flags.u.z : 4 debug bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The entry is used for debugging, currently holding an initial quadrant position code 
and 3 placeholder bytes.

optixrap/cu/generate.cu::

    491     // initial quadrant 
    492     uifchar4 c4 ;
    493     c4.uchar_.x =
    494                   (  p.position.x > 0.f ? QX : 0u )
    495                    |
    496                   (  p.position.y > 0.f ? QY : 0u )
    497                    |
    498                   (  p.position.z > 0.f ? QZ : 0u )
    499                   ;
    500 
    501     c4.uchar_.y = 2u ;   // 3-bytes up for grabs
    502     c4.uchar_.z = 3u ;
    503     c4.uchar_.w = 4u ;
    504 
    505     p.flags.f.z = c4.f ;
    506 
    507 


p.flags.u.w : photon history mask  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Opticks has both C++ and python machinery to interpret the history mask and analyse events

::

    In [4]: from opticks.ana.hismask import HisMask 

    In [5]: hm = HisMask()
    [2020-06-17 15:06:41,227] p70385 {__init__            :enum.py   :37} INFO     - parsing $OPTICKS_PREFIX/include/OpticksCore/OpticksPhoton.h 
    [2020-06-17 15:06:41,227] p70385 {__init__            :enum.py   :39} INFO     - path expands to /usr/local/opticks/include/OpticksCore/OpticksPhoton.h 

    In [6]: hm.label(65)
    Out[6]: 'SD|CK'

    In [7]: hm.label(1089)
    Out[7]: 'BR|SD|CK'

    In [8]: hm.abbrev.abbr2name
    Out[8]: 
    {'/usr/local/opticks/include/OpticksCore/OpticksFlags_Abbrev.json': 'jsonLoadPath',
     'AB': 'BULK_ABSORB',
     'BR': 'BOUNDARY_REFLECT',
     'BT': 'BOUNDARY_TRANSMIT',
     'CK': 'CERENKOV',
     'DR': 'SURFACE_DREFLECT',
     'FD': 'FABRICATED',
     'GN': 'G4GUN',
     'GS': 'GENSTEPSOURCE',
     'MI': 'MISS',
     'MY': 'MACHINERY',
     'NA': 'NAN_ABORT',
     'NL': 'NATURAL',
     'PS': 'PRIMARYSOURCE',
     'RE': 'BULK_REEMIT',
     'SA': 'SURFACE_ABSORB',
     'SC': 'BULK_SCATTER',
     'SD': 'SURFACE_DETECT',
     'SI': 'SCINTILLATION',
     'SO': 'EMITSOURCE',
     'SR': 'SURFACE_SREFLECT',
     'TO': 'TORCH',
     'XX': 'BAD_FLAG'}


For more detail on the content of Opticks Events and the meanings of the 
flags see *docs/opticks_event_data.rst*





ckm G4OpticksRecorder experiments
--------------------------------------

Expt with bringing in the monster (CFG4 OpticksEvent recording) by 
using G4OptickRecorder called from ckm: Run, Event, Track and Step actions eg:: 

   EventAction::BeginOfEventAction
   EventAction::EndOfEventAction





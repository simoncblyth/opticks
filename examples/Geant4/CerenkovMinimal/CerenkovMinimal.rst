CerenkovMinimal
==================

CerenkovMinimal.cc
    main, includes only G4.hh

G4
    holds instances of all the below action classes, PhysicsList<L4Cerenkov>
    and the Ctx. Connects them up. 
    
PhysicsList
    mostly standard but with templated Cerenkov class 

L4Cerenkov
    minimally modified Cerenkov process that invokes
    G4Opticks::GetOpticks()->collectCerenkovStep

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
    all the action ctors. Nexus. 
 
    Ctx::setTrack currently kills non-optical tracks after the first 
    genstep has been collected : for debugging with a single genstep 

    Ctx::setTrackOptical invokes G4Opticks::setAlignIndex(photon_record_id)
    Ctx::postTrackOptical invokes G4Opticks::setAlignIndex(-1) 


TrackInfo
    used to pass the photon_record_id from the L4Cerenkov photon generation 
    loop to subsequent propagation.    


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

OpHit



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
    309         opticks_photon_offset = G4Opticks::GetOpticks()->getNumPhotons();
    310         // total number of photons for all gensteps collected before this one
    311         // within this OpticksEvent (potentially crossing multiple G4Event) 
    312 
    313         G4Opticks::GetOpticks()->collectCerenkovStep(
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
    356         G4Opticks::GetOpticks()->setAlignIndex(record_id);
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
    468         G4Opticks::GetOpticks()->setAlignIndex(-1);
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
    486         G4Opticks::GetOpticks()->collectSecondaryPhotons(pParticleChange) ;
    487 #endif
    488 
    489         return pParticleChange;
    490 }










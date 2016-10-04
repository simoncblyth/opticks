Reemission Review
====================


How to proceed ?
------------------

* need to add DYB style reemission to CFG4 

First tack, teleport in the DsG4Scintillation code and try to get it to work::

    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.h .
    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc .
    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPhysConsOptical.h .



Adopting DYBOp into CFG4
---------------------------



flags borked, so flying blind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* lots of Undefined boundary status

tlaser.py::

      A:seqhis_ana      1:laser 
              8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                  4d        0.055            553       [2 ] TO AB
          cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012            122       [4 ] TO BT BT AB
             8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                 45d        0.006             65       [3 ] TO RE AB
              4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
            8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                455d        0.003             34       [4 ] TO RE RE AB
          cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
             8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
           8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
               4cccd        0.003             25       [5 ] TO BT BT BT AB
          cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                 46d        0.002             21       [3 ] TO SC AB
          cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
            4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                           10000         1.00 
       B:seqhis_ana     -1:laser 
                   0        0.850           8498       [1 ] ?0?
                  4d        0.071            708       [2 ] TO AB
                   d        0.028            276       [1 ] TO
                400d        0.017            168       [4 ] TO ?0? ?0? AB
              40000d        0.009             92       [6 ] TO ?0? ?0? ?0? ?0? AB
                  6d        0.008             82       [2 ] TO SC
                600d        0.004             35       [4 ] TO ?0? ?0? SC
                 46d        0.003             26       [3 ] TO SC AB
              60000d        0.002             16       [6 ] TO ?0? ?0? ?0? ?0? SC
               4000d        0.002             15       [5 ] TO ?0? ?0? ?0? AB
          400000000d        0.002             15       [10] TO ?0? ?0? ?0? ?0? ?0? ?0? ?0? ?0? AB
                 40d        0.001             11       [3 ] TO ?0? AB
            4000000d        0.001              7       [8 ] TO ?0? ?0? ?0? ?0? ?0? ?0? AB
             400600d        0.001              6       [7 ] TO ?0? ?0? SC ?0? ?0? AB
               4006d        0.001              6       [5 ] TO SC ?0? ?0? AB
          600000000d        0.001              6       [10] TO ?0? ?0? ?0? ?0? ?0? ?0? ?0? ?0? SC
             400006d        0.000              4       [7 ] TO SC ?0? ?0? ?0? ?0? AB
                 66d        0.000              3       [3 ] TO SC SC
               6006d        0.000              3       [5 ] TO SC ?0? ?0? SC
               6000d        0.000              3       [5 ] TO ?0? ?0? ?0? SC
                           10000         1.00 



live reemission photon counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STATIC buffer was expecting a certain number of photons, so currently truncates::

    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingAction@156] CSA (startEvent) event_id 9 event_total 9
    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100000 record_max 100000 STATIC 
    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100000 record_max 100000 STATIC 
    ...
    2016-10-04 11:49:42.529 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100495 record_max 100000 STATIC 
    2016-10-04 11:49:42.529 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100495 record_max 100000 STATIC 
    2016-10-04 11:49:42.532 INFO  [1669872] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1


Normally with fabricated (as opposed to G4 live) gensteps, the number of photons is known ahead of time.

Reemission means cannot know photon counts ahead of time ?

* that statement is true only if you count reemits are new photons, Opticks does not 
 

Contining the slot for reemiisions with G4 ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is necessary for easy comparisons between G4 and Opticks.

With Opticks a reemitted photon continues the lineage (buffer slot) 
of its predecessor but with G4 a fresh new particle is created ...  



REEMISSIONPROB is not a standard G4 property
----------------------------------------------

::

       +X horizontal tlaser from middle of DYB AD

       A: opticks, has reemission treatment aiming to match DYB NuWa DetSim 
                   (it is handled as a subset of BULK_ABSORB that confers rebirth)

       B: almost stock Geant4 10.2, no reemission treatment -> hence more absorption
                   (stock G4 is just absorbing, and the REEMISSIONPROB is ignored)


       A:seqhis_ana      1:laser 
              8ccccd        0.764         763501       [6 ] TO BT BT BT BT SA
                  4d        0.056          55825       [2 ] TO AB
          cccc9ccccd        0.025          25263       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.020          19707       [7 ] TO SC BT BT BT BT SA
                4ccd        0.013          12576       [4 ] TO BT BT AB
             8cccc5d        0.011          11183       [7 ] TO RE BT BT BT BT SA
              4ccccd        0.009           8554       [6 ] TO BT BT BT BT AB
                 45d        0.008           7531       [3 ] TO RE AB
            8cccc55d        0.005           5362       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004           4109       [7 ] TO BT BT SC BT BT SA
                455d        0.004           3588       [4 ] TO RE RE AB
             86ccccd        0.003           2836       [7 ] TO BT BT BT BT SC SA
          cccccc6ccd        0.003           2674       [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d        0.003           2524       [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd        0.002           2359       [7 ] TO BT BT RE BT BT SA
          cacccccc6d        0.002           2210       [10] TO SC BT BT BT BT BT BT SR BT
                 46d        0.002           2118       [3 ] TO SC AB
          cccc6ccccd        0.002           2060       [10] TO BT BT BT BT SC BT BT BT BT
               4cccd        0.002           1940       [5 ] TO BT BT BT AB
             89ccccd        0.002           1880       [7 ] TO BT BT BT BT DR SA
                         1000000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.813         813472       [6 ] TO BT BT BT BT SA
                  4d        0.072          71523       [2 ] TO AB
          cccc9ccccd        0.027          27170       [10] TO BT BT BT BT DR BT BT BT BT
                4ccd        0.017          17386       [4 ] TO BT BT AB
             8cccc6d        0.015          15107       [7 ] TO SC BT BT BT BT SA
              4ccccd        0.009           8842       [6 ] TO BT BT BT BT AB
          cacccccc6d        0.004           3577       [10] TO SC BT BT BT BT BT BT SR BT
             8cc6ccd        0.003           3466       [7 ] TO BT BT SC BT BT SA
                 46d        0.003           2515       [3 ] TO SC AB
             86ccccd        0.002           2476       [7 ] TO BT BT BT BT SC SA
           cac0ccc6d        0.002           2356       [9 ] TO SC BT BT BT ?0? BT SR BT
          cccccc6ccd        0.002           2157       [10] TO BT BT SC BT BT BT BT BT BT
             89ccccd        0.002           2127       [7 ] TO BT BT BT BT DR SA
               4cccd        0.002           1977       [5 ] TO BT BT BT AB
          cccc6ccccd        0.002           1949       [10] TO BT BT BT BT SC BT BT BT BT
            8ccccc6d        0.002           1515       [8 ] TO SC BT BT BT BT BT SA
          ccbccccc6d        0.001           1429       [10] TO SC BT BT BT BT BT BR BT BT
           4cc9ccccd        0.001           1215       [9 ] TO BT BT BT BT DR BT BT AB
                 4cd        0.001           1077       [3 ] TO BT AB
               4cc6d        0.001            802       [5 ] TO SC BT BT AB
                         1000000         1.00 



/usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.h::

    /// NB unlike stock G4  DsG4Scintillation::IsApplicable is true for opticalphoton
    ///    --> this is needed in order to handle the reemission of optical photons

    300 inline
    301 G4bool DsG4Scintillation::IsApplicable(const G4ParticleDefinition& aParticleType)
    302 {
    303         if (aParticleType.GetParticleName() == "opticalphoton"){
    304            return true;
    305         } else {
    306            return true;
    307         }
    308 }

    ///    NB the untrue comment, presumably inherited from stock G4 
    ///
    137         G4bool IsApplicable(const G4ParticleDefinition& aParticleType);
    138         // Returns true -> 'is applicable', for any particle type except
    139         // for an 'opticalphoton' 



/usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc::

    099 DsG4Scintillation::DsG4Scintillation(const G4String& processName,
    100                                      G4ProcessType type)
    101     : G4VRestDiscreteProcess(processName, type)
    102     , doReemission(true)
    103     , doBothProcess(true)
    104     , fPhotonWeight(1.0)
    105     , fApplyPreQE(false)
    106     , fPreQE(1.)
    107     , m_noop(false)
    108 {
    109     SetProcessSubType(fScintillation);
    110     fTrackSecondariesFirst = false;



    170 G4VParticleChange*
    171 DsG4Scintillation::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    172 
    173 // This routine is called for each tracking step of a charged particle
    174 // in a scintillator. A Poisson/Gauss-distributed number of photons is 
    175 // generated according to the scintillation yield formula, distributed 
    176 // evenly along the track segment and uniformly into 4pi.
    177 
    178 {
    179     aParticleChange.Initialize(aTrack);
    ...
    187     G4String pname="";
    188     G4ThreeVector vertpos;
    189     G4double vertenergy=0.0;
    190     G4double reem_d=0.0;
    191     G4bool flagReemission= false;

    193     if (aTrack.GetDefinition() == G4OpticalPhoton::OpticalPhoton()) 
            {
    194         G4Track *track=aStep.GetTrack();
    197 
    198         const G4VProcess* process = track->GetCreatorProcess();
    199         if(process) pname = process->GetProcessName();

    ///         flagReemission is set only for opticalphotons that are 
    ///         about to be bulk absorbed(fStopAndKill and !fGeomBoundary)
    ///
    ///           doBothProcess = true :  reemission for optical photons generated by both scintillation and Cerenkov processes         
    ///           doBothProcess = false : reemission for optical photons generated by Cerenkov process only 
    ///

    200 
    204         if(doBothProcess) 
               {
    205             flagReemission= doReemission
    206                 && aTrack.GetTrackStatus() == fStopAndKill
    207                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary;
    208         }
    209         else
                {
    210             flagReemission= doReemission
    211                 && aTrack.GetTrackStatus() == fStopAndKill
    212                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary
    213                 && pname=="Cerenkov";
    214         }
    218         if (!flagReemission) 
                {
    ///          -> give up the ghost and get absorbed
    219              return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    220         }
    221     }
    223     G4double TotalEnergyDeposit = aStep.GetTotalEnergyDeposit();
    228     if (TotalEnergyDeposit <= 0.0 && !flagReemission) {
    229         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    230     }
    ...
    246     if (aParticleName == "opticalphoton") {
    247       FastTimeConstant = "ReemissionFASTTIMECONSTANT";
    248       SlowTimeConstant = "ReemissionSLOWTIMECONSTANT";
    249       strYieldRatio = "ReemissionYIELDRATIO";
    250     }
    251     else if(aParticleName == "gamma" || aParticleName == "e+" || aParticleName == "e-") {
    252       FastTimeConstant = "GammaFASTTIMECONSTANT";
    ...
            }

    273     const G4MaterialPropertyVector* Fast_Intensity  = aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
    275     const G4MaterialPropertyVector* Slow_Intensity  = aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");
    277     const G4MaterialPropertyVector* Reemission_Prob = aMaterialPropertiesTable->GetProperty("REEMISSIONPROB");
    ...
    283     if (!Fast_Intensity && !Slow_Intensity )
    284         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    ...
    286     G4int nscnt = 1;
    287     if (Fast_Intensity && Slow_Intensity) nscnt = 2;
    ...
    291     G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
    292     G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    293 
    294     G4ThreeVector x0 = pPreStepPoint->GetPosition();
    295     G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
    296     G4double      t0 = pPreStepPoint->GetGlobalTime();
    297 
    298     //Replace NumPhotons by NumTracks
    299     G4int NumTracks=0;
    300     G4double weight=1.0;
    301     if (flagReemission) 
            {
    ...
    305         if ( Reemission_Prob == 0) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    307         G4double p_reemission= Reemission_Prob->GetProperty(aTrack.GetKineticEnergy());
    309         if (G4UniformRand() >= p_reemission) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    ////
    ////        above line reemission has a chance to not happen, otherwise we create a single secondary...
    ///         conferring reemission "rebirth"
    ////

    311         NumTracks= 1;
    312         weight= aTrack.GetWeight();
    316     else {
    317         //////////////////////////////////// Birks' law ////////////////////////






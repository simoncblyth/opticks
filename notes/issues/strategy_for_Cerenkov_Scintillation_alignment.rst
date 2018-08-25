strategy_for_Cerenkov_Scintillation_alignment
================================================

* for context see :doc:`OKG4Test_direct_two_executable_shakedown`

Realisation of need for strategy redirection came from comparing 
the simstreams (call locations of flat for ckm-- and ckm-okg4).  

Thoughts
-----------

Why was I trying to start from primaries ? Rather than gensteps ?

* because are trying to duplicate the G4 running too, and G4 cannot 
  start from gensteps 

  * yes G4 cannot start from Gensteps : but it can start from the photons that the gensteps yield 
 
    * **THIS IS THE CORE PROBLEM** are missing: CCerenkovSource (primary generator class)
      

  * original motivation is to try to align Cerenkov/Scintillation generation of photons  

  * need to find another approach, how to cleanroom isolate Cerenkov generation ?
    but Cerenkov needs the em energy loss to do anything  

    * how does the Cerenkov process interface to learn of the energy loss ? 
      Can I mock that up somehow  ? basically I need a clean environment 
      without a lot of physics calling flat  

* SO : need to focus first on matching the G4 runs, start by 
  collecting G4 hits into an OpticksEvent : not with full machinery 
  need a simple way, as want to do from 1st and 2nd executable  

  * just have a collector for hits too, hmm will need to do for 
    entire collection in reverse : so do at collection level ?
    then theres an intermediary hit class inbetween : better to 
    do in ProcessHits just like `CWriter::writePhoton(const G4StepPoint* point )`


Validating the C+S sources ?
-------------------------------

* simply compare the photons and hits from the natural full physics case
  with the cleanroom case booted from gensteps 
    

What is really needed ?
--------------------------

0. run full physics Geant4 : persist geometry and gensteps (Cerenkov and Scintillation) to geocache
1. run minimal physics Geant4 (cleanroom environment) : load geometry and gensteps 

   * generate some photons from them and propagate them
   * this is treating C+S processes as generators of photons, so they should not
     be included into the physics : just reuse the photon generation loop converting 
     the "secondaries" into "primaries"   

2. run Opticks : load geometry and gensteps 

   * generate some photons from them and propagate them 

3. compare 
   

* no hope of aligning 0 with anything other than another 0
* it may be possible to align 1 and 2 however 

* recall gensteps are the "stack" midway thru the Cerenkov and Scintillation processes 
  so getting a hacked version of these to generate photons from loaded gensteps should
  be straightforward 



Turning S+C into genstep consuming photon generators is entirely doable : but what then
------------------------------------------------------------------------------------------

Basically are extracting photon secondary formation from the C+S processes photon loops 
and turning them into generators (or Sources in Opticks lingo).

*  CCerenkovSource
*  CScintillationSource

Then this is something that can run with optical physics only : which 
is amenable to random alignment with Opticks which is doing the same thing
when running from gensteps. 


DONE : first cut CCerenkovGenerator + tests/CCerenkovGeneratorTest 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* standalone CCerenkovGeneratorTest executable

  * (requires loading full hub just to get at RINDEX for materials)
  * loads gensteps
  * generates and saves photons

* issues : with old gensteps had to force a materialIndex 

  * TODO: avoid by using new gensteps (from ckm)
  * TODO: wavelength units off by 1e6 : review unit conversions thru the chain



G4Cerenkov -> CCerenkovSource
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    168 G4VParticleChange*
    169 G4Cerenkov::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    170 
    171 // This routine is called for each tracking Step of a charged particle
    172 // in a radiator. A Poisson-distributed number of photons is generated
    173 // according to the Cerenkov formula, distributed evenly along the track
    174 // segment and uniformly azimuth w.r.t. the particle direction. The
    175 // parameters are then transformed into the Master Reference System, and
    176 // they are added to the particle change.
    177 
    178 {
    179   ////////////////////////////////////////////////////
    180   // Should we ensure that the material is dispersive?
    181   ////////////////////////////////////////////////////
    182 
    183   aParticleChange.Initialize(aTrack);
    184 
    ...
    ...
    255   G4double nMax = Rindex->GetMaxValue();
    256 
    257   G4double BetaInverse = 1./beta;
    258 
    259   G4double maxCos = BetaInverse / nMax;
    260   G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);
    261 
    262   G4double beta1 = pPreStepPoint ->GetBeta();
    263   G4double beta2 = pPostStepPoint->GetBeta();
    264 
    265   G4double MeanNumberOfPhotons1 =
    266                      GetAverageNumberOfPhotons(charge,beta1,aMaterial,Rindex);
    267   G4double MeanNumberOfPhotons2 =
    268                      GetAverageNumberOfPhotons(charge,beta2,aMaterial,Rindex);
    269 

    /////////  Gensteps persist the stack here   //////////  


    270   for (G4int i = 0; i < fNumPhotons; i++) {
    271 
    272       // Determine photon energy
    273 
    274       G4double rand;
    275       G4double sampledEnergy, sampledRI;
    276       G4double cosTheta, sin2Theta;
    277 
    278       // sample an energy
    279 
    280       do {
    281          rand = G4UniformRand();
    282          sampledEnergy = Pmin + rand * dp;
    283          sampledRI = Rindex->Value(sampledEnergy);
    284          cosTheta = BetaInverse / sampledRI;
    285 
    286          sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
    287          rand = G4UniformRand();
    288 
    289         // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
    290       } while (rand*maxSin2 > sin2Theta);
    291 
    ...
    368       G4Track* aSecondaryTrack =
    369                new G4Track(aCerenkovPhoton,aSecondaryTime,aSecondaryPosition);
    370 
    371       aSecondaryTrack->SetTouchableHandle(
    372                                aStep.GetPreStepPoint()->GetTouchableHandle());
    373 
    374       aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    375 
    376       aParticleChange.AddSecondary(aSecondaryTrack);
    377   }
    378 
    379   if (verboseLevel>0) {
    380      G4cout <<"\n Exiting from G4Cerenkov::DoIt -- NumberOfSecondaries = "
    381         << aParticleChange.GetNumberOfSecondaries() << G4endl;
    382   }
    383 
    384   return pParticleChange;
    385 }



aParticleChange vs pParticleChange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    282   protected:
    283       G4VParticleChange* pParticleChange;
    284       //  The pointer to G4VParticleChange object 
    285       //  which is modified and returned by address by the DoIt() method.
    286       //  This pointer should be set in each physics process
    287       //  after construction of derived class object.  
    288 
    289       G4ParticleChange aParticleChange;
    290       //  This object is kept for compatibility with old scheme
    291       //  This will be removed in future
    292 

     53 G4VProcess::G4VProcess(const G4String& aName, G4ProcessType   aType )
     54                   : aProcessManager(0),
     55                 pParticleChange(0),
     56                     theNumberOfInteractionLengthLeft(-1.0),
     57                     currentInteractionLength(-1.0),
     58             theInitialNumberOfInteractionLength(-1.0),
     59                     theProcessName(aName),
     60             theProcessType(aType),
     61             theProcessSubType(-1),
     62                     thePILfactor(1.0),
     63                     enableAtRestDoIt(true),
     64                     enableAlongStepDoIt(true),
     65                     enablePostStepDoIt(true),
     66                     verboseLevel(0),
     67                     masterProcessShadow(0)
     68 
     69 {
     70   pParticleChange = &aParticleChange;
     71 }





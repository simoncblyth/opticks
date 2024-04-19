g4velocity
===========


1042 SetVelocity 
-------------------

::

    epsilon:geant4.10.04.p02 blyth$ g4-g SetVelocity 
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4Track.hh://   Add SetVelocityTableProperties                 02 Apr. 2011  H.Kurashige
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4Track.hh:   void     SetVelocity(G4double val);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4Track.hh:  static void SetVelocityTableProperties(G4double t_max, G4double t_min, G4int nbin);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4Track.icc:   inline void  G4Track::SetVelocity(G4double val)
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4Track.cc:void G4Track::SetVelocityTableProperties(G4double t_max, G4double t_min, G4int nbin)
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4Track.cc:  G4VelocityTable::SetVelocityTableProperties(t_max, t_min, nbin);


    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4VelocityTable.hh:  static void SetVelocityTableProperties(G4double t_max, 

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4Step.icc:    fpPreStepPoint->SetVelocity(fpTrack->CalculateVelocity());
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4Step.icc:   fpTrack->SetVelocity(fpPostStepPoint->GetVelocity());

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4StepPoint.hh:   void SetVelocity(G4double v);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4StepPoint.icc: void G4StepPoint::SetVelocity(G4double v)

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4VelocityTable.cc:void G4VelocityTable::SetVelocityTableProperties(G4double t_max, G4double t_min, G4int nbin)
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4VelocityTable.cc:    G4Exception("G4VelocityTable::SetVelocityTableProperties",

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChangeForGamma.cc:      pPostStepPoint->SetVelocity(pTrack->CalculateVelocity());
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChangeForTransport.cc:  if (isVelocityChanged)  pPostStepPoint->SetVelocity(theVelocityChange);

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChange.cc:  pPostStepPoint->SetVelocity(theVelocityChange);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChange.cc:  pPostStepPoint->SetVelocity(theVelocityChange);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChange.cc:  pPostStepPoint->SetVelocity(theVelocityChange);

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChangeForLoss.cc:    pPostStepPoint->SetVelocity(0.0);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChangeForLoss.cc:    pPostStepPoint->SetVelocity(pTrack->CalculateVelocity());
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChangeForLoss.cc:    pPostStepPoint->SetVelocity(pTrack->CalculateVelocity());
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/src/G4ParticleChangeForLoss.cc:    pPostStepPoint->SetVelocity(0.0);

    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/solidstate/phonon/src/G4VPhononProcess.cc:  sec->SetVelocity(theLattice->MapKtoV(polarization, waveVec));    
    epsilon:geant4.10.04.p02 blyth$ 


1042 ProposeVelocity 
----------------------

::

    epsilon:geant4.10.04.p02 blyth$ g4-g ProposeVelocity 
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4ParticleChange.hh://   Add  Get/ProposeVelocity                       Apr 2011 H.Kurashige
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4ParticleChange.hh:    void ProposeVelocity(G4double finalVelocity);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/track/include/G4ParticleChange.icc:  void G4ParticleChange::ProposeVelocity(G4double finalVelocity)
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/hadronic/processes/src/G4UCNBoundaryProcess.cc:  aParticleChange.ProposeVelocity(aTrack.GetVelocity());
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/hadronic/processes/src/G4UCNBoundaryProcess.cc:          aParticleChange.ProposeVelocity(std::sqrt(2*Enew/neutron_mass_c2)*c_light);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/hadronic/processes/src/G4UCNBoundaryProcess.cc:          aParticleChange.ProposeVelocity(std::sqrt(2*Enew/neutron_mass_c2)*c_light);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/electromagnetic/dna/management/src/G4ITTransportation.cc:    fParticleChange.ProposeVelocity(initialVelocity);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/electromagnetic/dna/management/src/G4ITTransportation.cc:      fParticleChange.ProposeVelocity(finalVelocity);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/optical/src/G4OpBoundaryProcess.cc:        aParticleChange.ProposeVelocity(aTrack.GetVelocity());
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/optical/src/G4OpBoundaryProcess.cc:           aParticleChange.ProposeVelocity(finalVelocity);
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02/source/processes/solidstate/phonon/src/G4PhononReflection.cc:    aParticleChange.ProposeVelocity(vg);
    epsilon:geant4.10.04.p02 blyth$ 


1042 G4OpBoundaryProcess
-------------------------- 

::

     169 G4VParticleChange*
     170 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     171 {
     172         theStatus = Undefined;
     173 
     174         aParticleChange.Initialize(aTrack);
     175         aParticleChange.ProposeVelocity(aTrack.GetVelocity());
     ...
     542         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     543            G4MaterialPropertyVector* groupvel =
     544            Material2->GetMaterialPropertiesTable()->GetProperty(kGROUPVEL);
     545            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     546            aParticleChange.ProposeVelocity(finalVelocity);
     547         }
     548 
     549         if ( theStatus == Detection && fInvokeSD ) InvokeSD(pStep);
     550 
     551         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     552 }



1121 G4OpBoundaryProcess : additional ProposeVelocity for StepTooSmall using groupvel from fMaterial2
-------------------------------------------------------------------------------------------------------

* HMM: if StepTooSmall happens before reflection turnaround then fMaterial2 GROUPVEL would be the wrong one
* SO : HOW DOES THE TURNAROUND HAPPEN

  * G4OpBoundaryProcess::DoReflection JUST CHANGES mom,pol,status

* Fri Feb 16 14:58:45 2024 +0100 Import Geant4 11.2.1 source tree

::

     145 //....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
     146 G4VParticleChange* G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack,
     147                                                      const G4Step& aStep)
     148 {
     149   fStatus = Undefined;
     150   aParticleChange.Initialize(aTrack);
     151   aParticleChange.ProposeVelocity(aTrack.GetVelocity());

     ...
     186   G4double stepLength = aTrack.GetStepLength();
     187   if(stepLength <= fCarTolerance)
     188   {
     189     fStatus = StepTooSmall;
     190     if(verboseLevel > 1)
     191       BoundaryProcessVerbose();
     192 
     193     G4MaterialPropertyVector* groupvel = nullptr;
     194     G4MaterialPropertiesTable* aMPT = fMaterial2->GetMaterialPropertiesTable();
     195     if(aMPT != nullptr)
     196     {
     197       groupvel = aMPT->GetProperty(kGROUPVEL);
     198     }
     199 
     200     if(groupvel != nullptr)
     201     {
     202       aParticleChange.ProposeVelocity(
     203         groupvel->Value(fPhotonMomentum, idx_groupvel));
     204     }

     205     return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     206   }



Contrast with 1042::

     210         if (aTrack.GetStepLength()<=kCarTolerance/2){
     211                 theStatus = StepTooSmall;
     212                 if ( verboseLevel > 0) BoundaryProcessVerbose();
     213                 return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     214         }




     ...
     543   if(fStatus == FresnelRefraction || fStatus == Transmission)
     544   {
     545     // not all surface types check that fMaterial2 has an MPT
     546     G4MaterialPropertiesTable* aMPT = fMaterial2->GetMaterialPropertiesTable();
     547     G4MaterialPropertyVector* groupvel = nullptr;
     548     if(aMPT != nullptr)
     549     {
     550       groupvel = aMPT->GetProperty(kGROUPVEL);
     551     }
     552     if(groupvel != nullptr)
     553     {
     554       aParticleChange.ProposeVelocity(
     555         groupvel->Value(fPhotonMomentum, idx_groupvel));
     556     }
     557   }
     558 
     559   if(fStatus == Detection && fInvokeSD)
     560     InvokeSD(pStep);
     561   return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     562 }



1042 -> 1120 StepTooSmall change
----------------------------------

* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2438



1042 : Desc interplay between below classes for velocity 
-----------------------------------------------------------

G4Step::

   185  void G4Step::InitializeStep( G4Track* aValue )
   219     fpPreStepPoint->SetVelocity(fpTrack->CalculateVelocity());
   221    (*fpPostStepPoint) = (*fpPreStepPoint);
      
   * G4Step::InitializeStep fpPreStepPoint velocity from G4Track::CalculateVelocity
   * fpPreStepPoint copied to fpPostStepPoint

   225  void G4Step::UpdateTrack( )
   251    fpTrack->SetVelocity(fpPostStepPoint->GetVelocity());

   * G4Step::UpdateTrack velocity passed from fpPostStepPoint to fpTrack

G4StepPoint
   
   * acts as the dumb holder of velocity  

G4ParticleChange

    264 G4Step* G4ParticleChange::UpdateStepForAlongStep(G4Step* pStep)
    321   pPostStepPoint->SetVelocity(theVelocityChange);

    * pPostStepPoint velocity set for all three 

G4Track



1042 G4Step
-------------

::

    184 inline
    185  void G4Step::InitializeStep( G4Track* aValue )
    186  { 
    ...
    217    // Set Velocity
    218    //  should be placed after SetMaterial for preStep point 
    219     fpPreStepPoint->SetVelocity(fpTrack->CalculateVelocity());
    220    
    221    (*fpPostStepPoint) = (*fpPreStepPoint);
    222  }

    /// called from "void G4SteppingManager::SetInitialStep(G4Track* valueTrack)"


    223
    224 inline
    225  void G4Step::UpdateTrack( )
    226  {
    227    // To avoid the circular dependency between G4Track, G4Step
    228    // and G4StepPoint, G4Step has to manage the update actions.
    229    //  position, time
    230    fpTrack->SetPosition(fpPostStepPoint->GetPosition());
    231    fpTrack->SetGlobalTime(fpPostStepPoint->GetGlobalTime());
    232    fpTrack->SetLocalTime(fpPostStepPoint->GetLocalTime());
    233    fpTrack->SetProperTime(fpPostStepPoint->GetProperTime());
    234    //  energy, momentum, polarization
    235    fpTrack->SetMomentumDirection(fpPostStepPoint->GetMomentumDirection());
    236    fpTrack->SetKineticEnergy(fpPostStepPoint->GetKineticEnergy());
    237    fpTrack->SetPolarization(fpPostStepPoint->GetPolarization());
    238    //  mass charge
    239    G4DynamicParticle* pParticle = (G4DynamicParticle*)(fpTrack->GetDynamicParticle());
    240    pParticle->SetMass(fpPostStepPoint->GetMass());
    241    pParticle->SetCharge(fpPostStepPoint->GetCharge());
    242    //  step length
    243    fpTrack->SetStepLength(fStepLength);
    244    // NextTouchable is updated
    245    // (G4Track::Touchable points touchable of Pre-StepPoint)
    246    fpTrack->SetNextTouchableHandle(fpPostStepPoint->GetTouchableHandle());
    247    fpTrack->SetWeight(fpPostStepPoint->GetWeight());
    248 
    249 
    250    // set velocity
    251    fpTrack->SetVelocity(fpPostStepPoint->GetVelocity());
    252 }



1042 G4ParticleChange
-----------------------


::


    264 G4Step* G4ParticleChange::UpdateStepForAlongStep(G4Step* pStep)
    265 {
    266   // A physics process always calculates the final state of the
    267   // particle relative to the initial state at the beginning
    268   // of the Step, i.e., based on information of G4Track (or
    269   // equivalently the PreStepPoint). 
    270   // So, the differences (delta) between these two states have to be
    271   // calculated and be accumulated in PostStepPoint. 
    272  
    273   // Take note that the return type of GetMomentumDirectionChange is a
    274   // pointer to G4ParticleMometum. Also it is a normalized 
    275   // momentum vector.


    311   // calculate velocity
    312   if (!isVelocityChanged) {
    313     if(energy > 0.0) {
    314       pTrack->SetKineticEnergy(energy);
    315       theVelocityChange = pTrack->CalculateVelocity();
    316       pTrack->SetKineticEnergy(preEnergy);
    317     } else if(theMassChange > 0.0) {
    318       theVelocityChange = 0.0;
    319     }
    320   }
    321   pPostStepPoint->SetVelocity(theVelocityChange);

    344   //  Update the G4Step specific attributes 
    345   return UpdateStepInfo(pStep);
    346 }


    348 G4Step* G4ParticleChange::UpdateStepForPostStep(G4Step* pStep)
    349 { 
    350   // A physics process always calculates the final state of the particle
    351 
    352   // Take note that the return type of GetMomentumChange is a
    353   // pointer to G4ParticleMometum. Also it is a normalized 
    354   // momentum vector.
    ...
    368   // calculate velocity
    369   pTrack->SetKineticEnergy( theEnergyChange );
    370   if (!isVelocityChanged) {
    371     if(theEnergyChange > 0.0) {
    372       theVelocityChange = pTrack->CalculateVelocity();
    373     } else if(theMassChange > 0.0) {
    374       theVelocityChange = 0.0;
    375     }
    376   }
    377   pPostStepPoint->SetVelocity(theVelocityChange);


    402 G4Step* G4ParticleChange::UpdateStepForAtRest(G4Step* pStep)
    403 {
    404   // A physics process always calculates the final state of the particle
    405 
    ...
    415   pPostStepPoint->SetKineticEnergy( theEnergyChange );
    416   if (!isVelocityChanged) theVelocityChange = pStep->GetTrack()->CalculateVelocity();
    417   pPostStepPoint->SetVelocity(theVelocityChange);
    418 
    419   // update polarization



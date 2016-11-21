Geant4 Optical Code History
============================


Geant4 On Github
------------------

Dates from the patch files or release notes html

* https://github.com/Geant4/geant4/releases?after=v10.0.3

* v9.5.2 : Geant4 9.5 - patch-02 22 October 2012
* v9.5.1 : Geant4 9.5 - patch-01 20 March 2012
* v9.5.0 : release notes html date Dec 2nd 2011 

* https://github.com/Geant4/geant4/releases?after=v9.5.1



ProposeVelocity does nothing as CalculateVelocityForOpticalPhoton is repeatedly called
-----------------------------------------------------------------------------------------


GROUPVEL From Wrong Material Issue ?
---------------------------------------

Suspect seeing G4 bug that is fixed in lastest G4 with the below special case GROUPVEL access for

* dont think it was fixed anymore... unless there is incompatibility with my usage


/usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc::

     165 G4VParticleChange*
     166 G4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
     167 {
     ...
     529 
     530         aParticleChange.ProposeMomentumDirection(NewMomentum);
     531         aParticleChange.ProposePolarization(NewPolarization);
     532 
     533         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     534            G4MaterialPropertyVector* groupvel =
     535            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     536            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     537            aParticleChange.ProposeVelocity(finalVelocity);
     538         }
     539 
     540         if ( theStatus == Detection ) InvokeSD(pStep);
     541 
     542         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     543 }

Looking for the bug that induced the above special case, yeilds zilch.

* https://bugzilla-geant4.kek.jp/buglist.cgi?component=processes%2Foptical&product=Geant4

Found it

* https://github.com/mortenpi/geant4/blob/master/source/processes/optical/History

::

    24th Jan 2012 P.Gumplinger (op-V09-05-00)
                  G4OpBoundaryProcess.cc - solves Problem #1275
                  call aParticleChange.ProposeVelocity(aTrack.GetVelocity())
                  at every invocation of DoIt; for FresnelRefraction calculate
                  finalVelocity locally from Material2->
                  GetMaterialPropertiesTable()->GetProperty("GROUPVEL")

    28th Oct 2011 P.Gumplinger (op-V09-04-03)
                  add logic for ProposeVelocity to G4OpBoundaryProcess::PostStepDoIt






Bug 1275 : The velocity is wrong in optical photon propagation 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/420.html

Jan 2012 : Optical photons propagating at c instead of group velocity::

    I recently upgraded to GEANT 4.9.5 and have noticed that optical photons start to propagate 
    at c instead of the group velocity after a boundary process.  I didn't see this problem in 4.9.3.
    ...

Didnt find this bug initially as not tagged "optical"

* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=1275


Comparing this "patch" from the bug attachment  ~/Downloads/G4OpBoundaryProcess.cc with v9.5.0::

    simon:geant4-9.5.0 blyth$ diff $(g4dev-find) ~/Downloads/G4OpBoundaryProcess.cc 
    150a151
    >         aParticleChange.ProposeVelocity(aTrack.GetVelocity());
    483c484,486
    <            G4double finalVelocity = aTrack.CalculateVelocityForOpticalPhoton();
    ---
    >            G4MaterialPropertyVector* groupvel =
    >            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    >            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
    simon:geant4-9.5.0 blyth$ 







Close look at Bug 1275
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    gum 2012-01-09 02:44:12 CET

    Version 9.5

    A G4 user reported this bug:

    http://hypernews.slac.stanford.edu/HyperNews/geant4/get/opticalphotons/420.html

    Looks like in 9.5:

    mat=fpTouchable->GetVolume()->GetLogicalVolume()->GetMaterial();
    groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    velocity = groupvel->Value(current_momentum);

    in G4Track::CalculateVelocityForOpticalPhoton() calculates the velocity
    of the PreStepPoint and not, as expected, that of the PostStepPoint.
    Thus, the velocity after refraction is wrong.

    But it gets worse!

    G4ParticleChange::UpdateStepForPostStep

    has this line in it:

    if (!isVelocityChanged)theVelocityChange =
    pStep->GetTrack()->CalculateVelocity();

    what this means is that every other type of optical photon step also
    triggers CalculateVelocityForOpticalPhoton.

    That includes the zero-step-length reallocation step. It too triggers
    CalculateVelocityForOpticalPhoton. This now calculates and sets the
    velocity of the outside medium.

    The attached macro shows this for vanilla /examples/novice/N06


geant4_10_02_p01::

    254 ///////////////////
    255 G4double G4Track::CalculateVelocityForOpticalPhoton() const
    256 ///////////////////
    257 {
    258    
    259   G4double velocity = c_light ;
    260  
    261 
    262   G4Material* mat=0;
    263   G4bool update_groupvel = false;
    264   if ( fpStep !=0  ){
    265     mat= this->GetMaterial();         //   Fix for repeated volumes
    266   }else{
    267     if (fpTouchable!=0){
    268       mat=fpTouchable->GetVolume()->GetLogicalVolume()->GetMaterial();
    269     }
    270   }
    271   // check if previous step is in the same volume
    272     //  and get new GROUPVELOCITY table if necessary 
    273   if ((mat != 0) && ((mat != prev_mat)||(groupvel==0))) {
    274     groupvel = 0;
    275     if(mat->GetMaterialPropertiesTable() != 0)
    276       groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    277     update_groupvel = true;
    278   }
    279   prev_mat = mat;
    280  
    281   if  (groupvel != 0 ) {
    282     // light velocity = c/(rindex+d(rindex)/d(log(E_phot)))
    283     // values stored in GROUPVEL material properties vector
    284     velocity =  prev_velocity;
    285    
    286     // check if momentum is same as in the previous step
    287     //  and calculate group velocity if necessary 
    288     G4double current_momentum = fpDynamicParticle->GetTotalMomentum();
    289     if( update_groupvel || (current_momentum != prev_momentum) ) {
    290       velocity =
    291     groupvel->Value(current_momentum);
    292       prev_velocity = velocity;
    293       prev_momentum = current_momentum;
    294     }
    295   }  
    296  
    297   return velocity ;
    298 }


::

    lldb) bt
    * thread #1: tid = 0x1775c7, 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001126b8730) const + 20 at G4Track.cc:264, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001126b8730) const + 20 at G4Track.cc:264
        frame #1: 0x0000000106aeff2a libG4track.dylib`G4Track::G4Track(this=0x00000001126b8730, apValueDynamicParticle=<unavailable>, aValueTime=<unavailable>, aValuePosition=<unavailable>) + 474 at G4Track.cc:93
        frame #2: 0x0000000104bfdb07 libG4event.dylib`G4PrimaryTransformer::GenerateSingleTrack(this=0x0000000110214b60, primaryParticle=0x00000001126b7f50, x0=<unavailable>, y0=<unavailable>, z0=<unavailable>, t0=<unavailable>, wv=1) + 1687 at G4PrimaryTransformer.cc:219
        frame #3: 0x0000000104bfd434 libG4event.dylib`G4PrimaryTransformer::GenerateTracks(this=0x0000000110214b60, primaryVertex=<unavailable>) + 404 at G4PrimaryTransformer.cc:110
        frame #4: 0x0000000104bfd27b libG4event.dylib`G4PrimaryTransformer::GimmePrimaries(this=0x0000000110214b60, anEvent=<unavailable>, trackIDCounter=<unavailable>) + 155 at G4PrimaryTransformer.cc:81
        frame #5: 0x0000000104be0475 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000011020cfc0, anEvent=<unavailable>) + 1189 at G4EventManager.cc:160
        frame #6: 0x0000000104b62611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000110001a60, i_event=0) + 49 at G4RunManager.cc:399
::

    (lldb) bt
    * thread #1: tid = 0x1775c7, 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001126b8840) const + 20 at G4Track.cc:264, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001126b8840) const + 20 at G4Track.cc:264
        frame #1: 0x0000000104c7f54d libG4tracking.dylib`G4Step::InitializeStep(this=0x000000011020d220, aValue=<unavailable>) + 509 at G4Step.icc:219
        frame #2: 0x0000000104c7f02c libG4tracking.dylib`G4SteppingManager::SetInitialStep(this=0x000000011020d090, valueTrack=<unavailable>) + 1468 at G4SteppingManager.cc:356
        frame #3: 0x0000000104c884a7 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000011020d050, apValueG4Track=<unavailable>) + 199 at G4TrackingManager.cc:89
        frame #4: 0x0000000104be0727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000011020cfc0, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #5: 0x0000000104b62611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x0000000110001a60, i_event=0) + 49 at G4RunManager.cc:399
::

    (lldb) bt
    * thread #1: tid = 0x17803a, 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264
        frame #1: 0x0000000106ae88cd libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x000000011017a128, pStep=0x0000000108ff8d10) + 141 at G4ParticleChange.cc:372
        frame #2: 0x0000000104c80e3c libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000108ff8b80, np=<unavailable>) + 76 at G4SteppingManager2.cc:533
        frame #3: 0x0000000104c80d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000108ff8b80) + 139 at G4SteppingManager2.cc:502
        frame #4: 0x0000000104c7e909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000108ff8b80) + 825 at G4SteppingManager.cc:209
        frame #5: 0x0000000104c88771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108ff8b40, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #6: 0x0000000104be0727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000108ff8ab0, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #7: 0x0000000104b62611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e54d060, i_event=0) + 49 at G4RunManager.cc:399
::

    (lldb) bt
    * thread #1: tid = 0x17803a, 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264
        frame #1: 0x0000000103e20a64 libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000011017b670, aTrack=0x00000001102a6fb0, aStep=0x0000000108ff8d10) + 292 at DsG4OpBoundaryProcess.cc:200
        frame #2: 0x0000000104c80e2b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000108ff8b80, np=<unavailable>) + 59 at G4SteppingManager2.cc:530
        frame #3: 0x0000000104c80d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000108ff8b80) + 139 at G4SteppingManager2.cc:502
        frame #4: 0x0000000104c7e909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000108ff8b80) + 825 at G4SteppingManager.cc:209
        frame #5: 0x0000000104c88771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108ff8b40, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #6: 0x0000000104be0727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000108ff8ab0, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #7: 0x0000000104b62611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e54d060, i_event=0) + 49 at G4RunManager.cc:399
::

    (lldb) bt
    * thread #1: tid = 0x17803a, 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264
        frame #1: 0x0000000106ae88cd libG4track.dylib`G4ParticleChange::UpdateStepForPostStep(this=0x0000000110179668, pStep=0x0000000108ff8d10) + 141 at G4ParticleChange.cc:372
        frame #2: 0x0000000104c80e3c libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000108ff8b80, np=<unavailable>) + 76 at G4SteppingManager2.cc:533
        frame #3: 0x0000000104c80d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000108ff8b80) + 139 at G4SteppingManager2.cc:502
        frame #4: 0x0000000104c7e909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000108ff8b80) + 825 at G4SteppingManager.cc:209
        frame #5: 0x0000000104c88771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108ff8b40, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #6: 0x0000000104be0727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000108ff8ab0, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #7: 0x0000000104b62611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e54d060, i_event=0) + 49 at G4RunManager.cc:399
::

    (lldb) bt
    * thread #1: tid = 0x17803a, 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000106af0ab4 libG4track.dylib`G4Track::CalculateVelocityForOpticalPhoton(this=0x00000001102a6fb0) const + 20 at G4Track.cc:264
        frame #1: 0x0000000103e20a64 libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000011017b670, aTrack=0x00000001102a6fb0, aStep=0x0000000108ff8d10) + 292 at DsG4OpBoundaryProcess.cc:200
        frame #2: 0x0000000104c80e2b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000108ff8b80, np=<unavailable>) + 59 at G4SteppingManager2.cc:530
        frame #3: 0x0000000104c80d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000108ff8b80) + 139 at G4SteppingManager2.cc:502
        frame #4: 0x0000000104c7e909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000108ff8b80) + 825 at G4SteppingManager.cc:209
        frame #5: 0x0000000104c88771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x0000000108ff8b40, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #6: 0x0000000104be0727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000108ff8ab0, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #7: 0x0000000104b62611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010e54d060, i_event=0) + 49 at G4RunManager.cc:399






Checking History
~~~~~~~~~~~~~~~~~~~~~

Try looking at code history

* http://www-geant4.kek.jp/lxr/source//processes/optical/src/G4OpBoundaryProcess.cc
* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=8.0  Not there
* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=9.5  Nope
* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=9.6  First appearance, for only FresnelRefraction

::

    497         if ( theStatus == FresnelRefraction ) {
    498            G4MaterialPropertyVector* groupvel =
    499            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    500            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
    501            aParticleChange.ProposeVelocity(finalVelocity);
    502         }

* http://www-geant4.kek.jp/lxr/source/processes/optical/src/G4OpBoundaryProcess.cc?v=10.1 Adds in Transmission

::

    532         if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
    533            G4MaterialPropertyVector* groupvel =
    534            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    535            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
    536            aParticleChange.ProposeVelocity(finalVelocity);
    537         }
    538 

Look for commit history, Geant4 svn is hidden behind CERN login, try mirrors.

The below have no history

* https://gitlab.cern.ch/geant4/geant4/commits/master/source/processes/optical/src/G4OpBoundaryProcess.cc
* https://github.com/alisw/geant4


Add to cfg4/DsG4OpBoundaryProcess.cc::

     600         
     601 #ifdef GEANT4_BT_GROUPVEL_FIX
     602     // from /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc
     603        if ( theStatus == FresnelRefraction || theStatus == Transmission ) {
     604            G4MaterialPropertyVector* groupvel =
     605            Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
     606            G4double finalVelocity = groupvel->Value(thePhotonMomentum);
     607            aParticleChange.ProposeVelocity(finalVelocity);
     608         }
     609 #endif  
     610 





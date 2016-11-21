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


GROUPVEL From Wrong Material Issue ?
---------------------------------------

Suspect seeing G4 bug that is fixed in lastest G4 with the below special case GROUPVEL access for


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





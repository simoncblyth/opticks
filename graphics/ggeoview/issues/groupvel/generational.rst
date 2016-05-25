Groupvel Kludge Unintended Consequences
===========================================

See *ggv-pmt-test-notes*.

Following the g4gun generalizations viz shows very different behaviour.
Very slow moving photons... 

* switching back to old PhysicsList shows same mis-behaviour 

* pmt_test.py indicate material/flag sequence histories are matching... points 
  to problem with time  

* running with --steppingdbg shows the photons are moving very slowly 
  taking 200ns to get to the PMT.  What was that groupvel kludge ? 

* pmt_test_distrib.py shows problem is with G4 times, looks like always getting smth close to -10 for other than 1st::

    ===================================== ===== ===== ===== ======== ===== ===== ===== ===== 
    4/PmtInBox/torch : 107598/107251  :   X     Y     Z     T        A     B     C     R     
    ===================================== ===== ===== ===== ======== ===== ===== ===== ===== 
    [TO] BT SD                             0.91  0.73  0.56  0.56     0.98  1.09  0.56  0.94 
    TO [BT] SD                             0.91  0.73  0.81 11936.06  0.98  1.09  0.56  0.94 
    TO BT [SD]                             1.00  0.83  0.97 9341.26   0.98  1.09  0.56  0.93 
    ===================================== ===== ===== ===== ======== ===== ===== ===== ===== 

::

    (lldb) b "G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&)" 

    (lldb) b 537
    Breakpoint 2: where = libG4processes.dylib`G4OpBoundaryProcess::PostStepDoIt(G4Track const&, G4Step const&) + 8675 at G4OpBoundaryProcess.cc:537, address = 0x00000001042b7123
    (lldb) c
       ...
       534             G4MaterialPropertyVector* groupvel =
       535             Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
       536             G4double finalVelocity = groupvel->Value(thePhotonMomentum);
    -> 537             aParticleChange.ProposeVelocity(finalVelocity);
       538          }
       539  
    (lldb) p finalVelocity
    (G4double) $0 = 0.99930819333333331

    ## huh : speed of light should be ~300 mm/ns

    (lldb) p groupvel
    (G4MaterialPropertyVector *) $1 = 0x00000001091d3bc0
    (lldb) p *groupvel
    (G4MaterialPropertyVector) $2 = {
      G4PhysicsVector = {
        type = T_G4PhysicsOrderedFreeVector
        edgeMin = 0.0000015120022870975581
        edgeMax = 0.000020664031256999959
        numberOfNodes = 39
        dataVector = size=39 {
          [0] = 0.99930819333333331
          [1] = 0.99930819333333331
          [2] = 0.99930819333333331
          [3] = 0.99930819333333331
          [4] = 0.99930819333333331
          [5] = 0.99930819333333331
          [6] = 0.99930819333333331
          [7] = 0.99930819333333331
          [8] = 0.99930819333333331



Disabling the groupvel kludge via testconfig leads to G4ParticleChange not OK for velocity
and loadsa output. This suggests RINDEX is messed up ?

* the disabling failed to do so, as the former kludged GROUPVEL was 
  living as ggeo "groupvel" properties in the geocache and got sprung into 
  G4 materials by CPropLib 

* also somehow (maybe related to no-name extra props) RINDEX was stomped on 
  and got crazy low GROUPVEL of 0.999308193333
 
Check the GROUPVEL calc::

    (lldb) b "G4Track::CalculateVelocityForOpticalPhoton() const" 





GROUPVEL
==========

Approach
-----------


Issue 1
--------

This is a longstanding issue of proagation time mismatch between Opticks and G4

Longterm solution is to export GROUPVEL property together with 
RINDEX and others in the G4DAE export.  Not prepared to go there
yet though, so need a shortcut way to get this property into the
Opticks boundary texture.

Issue 2
-------

Recording step times after proposed velocity has a chance
to take effect.



See Also
---------

* ana/groupvel.py 
* ggeo/GProperty<T>::make_GROUPVEL


GROUPVEL Injection at tail of PSDIP has no effect
---------------------------------------------------

Looks like PSDIP proposed velocities never make it into transport calc::

    2016-11-21 18:34:23.260 INFO  [1509817] [CTrackingAction::setPhotonId@125] CTrackingAction::setPhotonId track_id 1 parent_id -1 primary_id -1 photon_id 1 reemtrack 0
    2016-11-21 18:34:23.260 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]  -1     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.260 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]  -1    bndary.PSDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   0     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.end wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   1     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [*DsG4OpBoundaryProcess::PostStepDoIt@738] inject Bialkali groupvel 205.619 at step_id 1
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.end wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   2     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.end wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   3     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   3    bndary.PSDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   3    bndary.PSDIP.end wavelength   430 groupvel    197.134 lookupMat MineralOil
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   4     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   4    bndary.PSDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CTrackingAction::setPhotonId@123] .
    2016-11-21 18:34:23.261 INFO  [1509817] [CTrackingAction::setPhotonId@124] .
    2016-11-21 18:34:23.261 INFO  [1509817] [CTrackingAction::setPhotonId@125] CTrackingAction::setPhotonId track_id 0 parent_id -1 primary_id -2 photon_id 0 reemtrack 0
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   5     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   5    bndary.PSDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   5    bndary.PSDIP.end wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   0     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.end wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   1     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.261 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.262 INFO  [1509817] [*DsG4OpBoundaryProcess::PostStepDoIt@738] inject Bialkali groupvel 205.619 at step_id 1
    2016-11-21 18:34:23.262 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.end wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:34:23.262 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   2     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.262 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:34:23.262 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.end wavelength   430 groupvel    197.134 lookupMat MineralOil
    2016-11-21 18:34:23.262 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   3     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.262 INFO  [1509817] [CMaterialLib::dumpGroupvelMaterial@38]   3    bndary.PSDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:34:23.262 INFO  [1509817] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2016-11-21 18:34:23.262 INFO  [1509817] [CG4::postpropagate@336] CG4::postpropagate(0)

::

    2016-11-21 18:42:01.307 INFO  [1512715] [CTrackingAction::setPhotonId@125] CTrackingAction::setPhotonId track_id 1 parent_id -1 primary_id -1 photon_id 1 reemtrack 0
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]  -1     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.307 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id -1
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]  -1    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   0     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.307 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 0
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.end wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   1     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.307 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 1
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.end wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   2     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.307 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 2
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.end wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   3     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.307 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 3
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   3    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   3    bndary.PSDIP.end wavelength   430 groupvel    197.134 lookupMat MineralOil
    2016-11-21 18:42:01.307 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   4     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.308 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 4
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   4    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.308 INFO  [1512715] [CTrackingAction::setPhotonId@123] .
    2016-11-21 18:42:01.308 INFO  [1512715] [CTrackingAction::setPhotonId@124] .
    2016-11-21 18:42:01.308 INFO  [1512715] [CTrackingAction::setPhotonId@125] CTrackingAction::setPhotonId track_id 0 parent_id -1 primary_id -2 photon_id 0 reemtrack 0
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   5     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.308 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 5
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   5    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   5    bndary.PSDIP.end wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   0     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.308 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 0
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   0    bndary.PSDIP.end wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   1     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.308 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 1
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   1    bndary.PSDIP.end wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   2     trans.ASDIP.beg wavelength   430 groupvel    194.519 lookupMat GdDopedLS
    2016-11-21 18:42:01.308 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 2
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   2    bndary.PSDIP.end wavelength   430 groupvel    197.134 lookupMat MineralOil
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   3     trans.ASDIP.beg wavelength   430 groupvel     192.78 lookupMat Acrylic
    2016-11-21 18:42:01.308 INFO  [1512715] [*DsG4OpBoundaryProcess::PostStepDoIt@182] inject Bialkali groupvel startVelocity 205.619 at step_id 3
    2016-11-21 18:42:01.308 INFO  [1512715] [CMaterialLib::dumpGroupvelMaterial@38]   3    bndary.PSDIP.beg wavelength   430 groupvel    205.619 lookupMat Bialkali
    2016-11-21 18:42:01.308 INFO  [1512715] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1






Can I replace G4Transportation with a debug version ?
--------------------------------------------------------

cfg4::

    154 void OpNovicePhysicsList::ConstructProcess()
    155 {
    156   setupEmVerbosity(0);
    157 
    158   AddTransportation();
    159   ConstructDecay();
    160   ConstructEM();
    161 
    162   ConstructOpDYB();
    163 
    164   dump("OpNovicePhysicsList::ConstructProcess");
    165 }


AddTransportation
~~~~~~~~~~~~~~~~~~~

::

    simon:cfg4 blyth$ g4-cc AddTransportation
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4PhysicsListHelper.cc:void G4PhysicsListHelper::AddTransportation()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4PhysicsListHelper.cc:    G4cout << "G4PhysicsListHelper::AddTransportation()  "<< G4endl;
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4PhysicsListHelper.cc:      G4cout << " G4PhysicsListHelper::AddTransportation()"
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4PhysicsListHelper.cc: G4cout << "G4PhysicsListHelper::AddTransportation  "
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4PhysicsListHelper.cc:      G4Exception("G4PhysicsListHelper::AddTransportation",
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4VModularPhysicsList.cc: AddTransportation();
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4VUserPhysicsList.cc:void G4VUserPhysicsList::AddTransportation()
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/run/src/G4VUserPhysicsList.cc:  G4MT_thePLHelper->AddTransportation();

::

     956 void G4VUserPhysicsList::AddTransportation()
     957 {
     958   G4MT_thePLHelper->AddTransportation();
     959 }





::

    simon:cfg4 blyth$ g4-cc G4Transportation\(\)
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/error_propagation/src/G4ErrorPhysicsList.cc:  G4Transportation* theTransportationProcess= new G4Transportation();
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/transportation/src/G4Transportation.cc:G4Transportation::~G4Transportation()



    111 void G4ErrorPhysicsList::ConstructProcess()
    112 {
    113   G4Transportation* theTransportationProcess= new G4Transportation();
    114 
    115 #ifdef G4VERBOSE
    116     if (verboseLevel >= 4){
    117       G4cout << "G4VUserPhysicsList::ConstructProcess()  "<< G4endl;
    118     }
    119 #endif
    120 
    121   // loop over all particles in G4ParticleTable
    122   theParticleIterator->reset();
    123   while( (*theParticleIterator)() ) {  // Loop checking, 06.08.2015, G.Cosmo
    124     G4ParticleDefinition* particle = theParticleIterator->value();
    125     G4ProcessManager* pmanager = particle->GetProcessManager();
    126     if (!particle->IsShortLived()) {
    127       G4cout << particle << "G4ErrorPhysicsList:: particle process manager " << particle->GetParticleName() << " = " << particle->GetProcessManager() << G4endl;
    128       // Add transportation process for all particles other than  "shortlived"
    129       if ( pmanager == 0) {
    130         // Error !! no process manager
    131         G4String particleName = particle->GetParticleName();
    132         G4Exception("G4ErrorPhysicsList::ConstructProcess","No process manager",
    133                     RunMustBeAborted, particleName );
    134       } else {
    135         // add transportation with ordering = ( -1, "first", "first" )
    136         pmanager ->AddProcess(theTransportationProcess);
    137         pmanager ->SetProcessOrderingToFirst(theTransportationProcess, idxAlongStep);
    138         pmanager ->SetProcessOrderingToFirst(theTransportationProcess, idxPostStep);
    139       }
    140     } else {
    141       // shortlived particle case
    142     }
    143   }






DsG4OpBoundaryProcess dumping : looks like getting groupvel from Ac instead of LS and MO
-------------------------------------------------------------------------------------------

tconcentric-i::

    In [2]: ab.b.sel = "TO BT BT BT BT SA"

    In [6]: ab.b.psel_dindex(slice(0,100))     # first 100 of top line, straight thrus (easy to interpret)
    Out[6]: '--dindex=1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,19,20,23,25,27,29,31,35,36,37,38,39,40,41,42,43,47,48,49,50,52,55,58,60,61,67,72,73,74,75,76,78,79,80,82,86,87,89,93,94,95,96,97'


In [1]: ab.b.psel_dindex(limit=10, reverse=True)
Out[1]: '--dindex=999999,999997,999996,999995,999994,999993,999992,999991,999990,999989'




Back to basics after moving to fine domain (1nm)
--------------------------------------------------

::

   tconcentric-tt --finedbndtex


Fine domain means can no longer blame interpolation mismatch for discreps

::
 
                    |
                    | 3000             4000             5000
         0          | + |               +                +
         +          |   |             |   |            |   |
        TO         BT   BT            BT  BT           SA  
              0     | 1 |      2      | 3 |     4      |   |
                    |   |             |   |            |   | 
                    |   |             |   |            |   | 
                    |   |             |   |            |   | 
                    |   |             |   |            |   | 

Calculate expectations for global times with tconcentric geometry, in bnd.py::

    Gd,LS,Ac,MO = 0,1,2,3
    gvel = i1m.data[(Gd,Ac,LS,Ac,MO),1,430-60,0]
    dist = np.array([0,3000-5,3000+5,4000-5,4000+5,5000-5], dtype=np.float32)   # tconcentric radii
    ddif = np.diff(dist)
    tdif = ddif/gvel
    tabs = np.cumsum(ddif/gvel) + 0.1 

    print "gvel: %r " %  gvel
    print "dist: %r " %  dist
    print "ddif: %r " %  ddif
    print "tdif: %r " %  tdif
    print "tabs: %r " %  tabs

    // with correct groupvel material order : (Gd,Ac,LS,Ac,MO)  get the Opticks times

    gvel: array([ 194.5192,  192.7796,  194.5192,  192.7796,  197.1341], dtype=float32) 
    dist: array([    0.,  2995.,  3005.,  3995.,  4005.,  4995.], dtype=float32) 
    ddif: array([ 2995.,    10.,   990.,    10.,   990.], dtype=float32) 
    tdif: array([ 15.3969,   0.0519,   5.0895,   0.0519,   5.022 ], dtype=float32) 
    tabs: array([ 15.4969,  15.5488,  20.6383,  20.6902,  25.7121], dtype=float32) 

    // mangling groupvel material order to : (Gd,LS,Ac,MO,Ac) nearly reproduces the CFG4 times...

    gvel2: array([ 194.5192,  194.5192,  192.7796,  197.1341,  192.7796], dtype=float32) 
    tdif2: array([ 15.3969,   0.0514,   5.1354,   0.0507,   5.1354], dtype=float32) 
    tabs2: array([ 15.4969,  15.5483,  20.6837,  20.7345,  25.8699], dtype=float32) 

    // another mangle to (Gd,LS,Ac,LS,Ac) reproduces the CFG4 times

    gvel3: array([ 194.5192,  194.5192,  192.7796,  194.5192,  192.7796], dtype=float32) 
    tdif3: array([ 15.3969,   0.0514,   5.1354,   0.0514,   5.1354], dtype=float32) 
    tabs3: array([ 15.4969,  15.5483,  20.6837,  20.7352,  25.8706], dtype=float32) 








Hmm looks like difference between use of preVelocity vs postVelocity (are using pre when should be using post).
Potentially due to CRecorder operating PRE_SAVE ?

Hmm to simplify recording, maybe better to move to trajectory style. Collecting steps into a container
within the UserSteppingAction and recording them from the UserTrackingAction after all tracking is done.
See: G4TrackingManager::ProcessOneTrack

::

    202 void G4Trajectory::AppendStep(const G4Step* aStep)
    203 {
    204    positionRecord->push_back( new G4TrajectoryPoint(aStep->GetPostStepPoint()->
    205                                  GetPosition() ));
    206 }
    207 




::

    DsG4OpBoundaryProcess::PostStepDoIt step_id    0 nm        430 priorVelocity    194.519 groupvel_m1            GdDopedLS   194.519 groupvel_m2              Acrylic    192.78 <-proposed 
    DsG4OpBoundaryProcess::PostStepDoIt step_id    1 nm        430 priorVelocity    194.519 groupvel_m1              Acrylic    192.78 groupvel_m2   LiquidScintillator   194.519 <-proposed 
    DsG4OpBoundaryProcess::PostStepDoIt step_id    2 nm        430 priorVelocity     192.78 groupvel_m1   LiquidScintillator   194.519 groupvel_m2              Acrylic    192.78 <-proposed 
    DsG4OpBoundaryProcess::PostStepDoIt step_id    3 nm        430 priorVelocity    194.519 groupvel_m1              Acrylic    192.78 groupvel_m2           MineralOil   197.134 <-proposed 

    // proposed velocity look correct, but suspect the recording happens too soon to feel the effect of it due to PRE_SAVE ??


    CRecorder::RecordStep trackStepLength       2995 trackGlobalTime    15.4969 trackVelocity    194.519 preVelocity    194.519 postVelocity    194.519 preDeltaTime    15.3969 postDeltaTime    15.3969
    CRecorder::RecordStep trackStepLength         10 trackGlobalTime    15.5483 trackVelocity     192.78 preVelocity    194.519 postVelocity     192.78 preDeltaTime  0.0514088 postDeltaTime  0.0518727
    CRecorder::RecordStep trackStepLength        990 trackGlobalTime    20.6837 trackVelocity    194.519 preVelocity     192.78 postVelocity    194.519 preDeltaTime     5.1354 postDeltaTime    5.08947
    CRecorder::RecordStep trackStepLength         10 trackGlobalTime    20.7352 trackVelocity     192.78 preVelocity    194.519 postVelocity     192.78 preDeltaTime  0.0514088 postDeltaTime  0.0518727
    CRecorder::RecordStep trackStepLength        990 trackGlobalTime    25.8706 trackVelocity    197.134 preVelocity     192.78 postVelocity    197.134 preDeltaTime     5.1354 postDeltaTime    5.02196

::
 
     TO   
     BT   Gd/Ac
     BT   Ac/LS
     BT   LS/Ac
     BT   Ac/MO
     SA   MO/Ac





Caution heavy compression with below values::

    ab.sel = "TO BT BT BT BT [SA]"

    a,b = ab.rpost()

    In [42]: a[0]
    Out[42]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.6377],
           [ 4004.9776,     0.    ,     0.    ,    20.6901],
           [ 4995.0716,     0.    ,     0.    ,    25.7136]])

    In [43]: b[0]
    Out[43]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8707]])
    
    In [4]: b[0]   ## after adding BT ProposeVelocity for m2 ... huh why almost no difference 
    Out[4]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4934],
           [ 3004.9551,     0.    ,     0.    ,    15.5458],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8666]])



Post recording returns to the values without the BT proposeVelocity::

    In [4]: b[0]
    Out[4]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8707]])







::

    2016-11-19 14:23:15.001 INFO  [1049278] [CRec::dump@40] CRec::dump record_id 999989 nstp 5  Ori[ 0.0000.0000.000] 
    ( 0)  TO/BT     FrT                                 PRE_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys       GdDopedLS          noProc           Undefined pos[      0.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns  0.100 nm 430.000
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   2995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.497 nm 430.000
     )
    ( 1)  BT/BT     FrT                                            PRE_SAVE 
    [   1](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   2995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.497 nm 430.000
     post               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.548 nm 430.000
     )
    ( 2)  BT/BT     FrT                                            PRE_SAVE 
    [   2](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys uidScintillator  Transportation        GeomBoundary pos[   3005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 15.548 nm 430.000
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.684 nm 430.000
     )
    ( 3)  BT/BT     FrT                                            PRE_SAVE 
    [   3](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   3995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.684 nm 430.000
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.735 nm 430.000
     )
    ( 4)  BT/SA     Abs     PRE_SAVE POST_SAVE POST_DONE LAST_POST SURF_ABS 
    [   4](Stp ;opticalphoton stepNum -561600160(tk ;opticalphoton tid 9990 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ 4995.000   0.000   0.000]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   4005.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 20.735 nm 430.000
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   4995.000     0.000     0.000]  dir[    1.000   0.000   0.000]  pol[    0.000   1.000   0.000]  ns 25.871 nm 430.000
     )






    
    In [44]: b[0,:,0] == a[0,:,0]    ## 2 simulations yield precisely the same positions
    Out[44]: 
    A()sliced
    A([ True,  True,  True,  True,  True,  True], dtype=bool) 

    In [45]: b[0,:,3] == a[0,:,3]
    Out[45]: 
    A()sliced
    A([ True,  True,  True, False, False, False], dtype=bool)


    In [46]: b[0,:,3] - a[0,:,3]
    Out[46]: 
    A()sliced
    A([ 0.    ,  0.    ,  0.    ,  0.0443,  0.0443,  0.1571])    ## time offset starts in LS, Acrylic does not add to it, MO makes it worse


Group velocity tex props from GdLS,LS,Ac,MO around 430nm::


    In [113]: i1m.data[(0,1,2,3),1,429-60:432-60,0]
    Out[113]: 
    array([[ 194.4354,  194.5192,  194.603 ],
           [ 194.4354,  194.5192,  194.603 ],
           [ 192.6459,  192.7796,  192.9132],
           [ 197.0692,  197.1341,  197.1991]], dtype=float32)

    In [114]: i2m.data[(0,1,2,3),1,429-60:432-60,0]
    Out[114]: 
    array([[ 194.4354,  194.5192,  194.603 ],
           [ 194.4354,  194.5192,  194.603 ],
           [ 192.6459,  192.7796,  192.9132],
           [ 197.0692,  197.1341,  197.1991]], dtype=float32)



Distances, time deltas, velocities for each step::

    In [96]: np.diff( a[0,:,0] ), np.diff( b[0,:,0] )    ## mm
    Out[96]: 
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]),
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]))

    In [97]: np.diff( a[0,:,3] ), np.diff( b[0,:,3] )    ## ns 
    Out[97]: 
    A([ 15.3967,       0.0524,   5.0879,       0.0524,   5.0235]),
    A([ 15.3967,       0.0524,   5.1322,       0.0524,   5.1363]))

              ratio of diffs                  ## mm/ns
    A([ 194.5238,  189.5833,   194.5969,   189.5833,   197.0937]),
    A([ 194.5238,  189.5833,  *192.9167*,  189.5833,  *192.7654*]))

    ##   (TO BT)   (BT BT)     (BT BT)     (BT BT)     (BT SA)          

    ##   Gd         Ac           LS          Ac         MO
    ##
    ## Ac precision very limited due to short time,dist and deep compression ??
    ##
    ## CFG4 gvel numbers for LS and MO look wrong ...
    ##      in fact they look like the Ac numbers  
    ##  


::

    GEANT4_BT_GROUPVEL_FIX m1            GdDopedLS m2              Acrylic eV    2.88335 nm        430 finalVelocity     192.78 priorVelocity    194.519 finalVelocity_m1    194.519
    GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2   LiquidScintillator eV    2.88335 nm        430 finalVelocity    194.519 priorVelocity    194.519 finalVelocity_m1     192.78
    GEANT4_BT_GROUPVEL_FIX m1   LiquidScintillator m2              Acrylic eV    2.88335 nm        430 finalVelocity     192.78 priorVelocity     192.78 finalVelocity_m1    194.519
    GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2           MineralOil eV    2.88335 nm        430 finalVelocity    197.134 priorVelocity    194.519 finalVelocity_m1     192.78


Is there an issue with CRecorder recording the times during stepping before fully baked ?








After 1st try at applying GEANT4_BT_GROUPVEL_FIX minimal change, is there a material swap? that happens on DR?:

    In [5]: np.diff( a[0,:,0] ), np.diff( b[0,:,0] ), np.diff( a[0,:,3] ), np.diff( b[0,:,3] ), np.diff( a[0,:,0] )/np.diff( a[0,:,3] ), np.diff( b[0,:,0] )/np.diff( b[0,:,3] )
    Out[5]: 
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]),
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 ]),
    A([ 15.3967,   0.0524,   5.0879,   0.0524,   5.0235]),
    A([ 15.3927,   0.0524,   5.1363,   0.0524,   5.1322]),
    A([ 194.5238,  189.5833,  194.5969,  189.5833,  197.0937]),
    A([ 194.5747,  189.5833,  192.7654,  189.5833,  192.9167]))



::

    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1            GdDopedLS m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2   LiquidScintillator eV    2.88335 nm        430 gv    194.519
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1   LiquidScintillator m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2           MineralOil eV    2.88335 nm        430 gv    197.134
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1            GdDopedLS m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2   LiquidScintillator eV    2.88335 nm        430 gv    194.519
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1   LiquidScintillator m2              Acrylic eV    2.88335 nm        430 gv     192.78
    2016-11-19 11:39:16.947 INFO  [1002089] [*DsG4OpBoundaryProcess::PostStepDoIt@610] GEANT4_BT_GROUPVEL_FIX m1              Acrylic m2           MineralOil eV    2.88335 nm        430 gv    197.134






::

    In [117]: ab.sel = "TO BT BT BT BT [DR] SA"

    In [118]: a,b = ab.rpost()

    In [119]: a.shape, b.shape
    Out[119]: (7540, 7, 4),  (7677, 7, 4)

    In [123]: a[0]
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.6377],
           [ 4004.9776,     0.    ,     0.    ,    20.6901],
           [ 4995.0716,     0.    ,     0.    ,    25.7136],
           [ 2840.6014,  -320.0011,  4096.1664,    49.2437]])

    In [124]: b[0]
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8707],
           [ 3076.4399,  -722.179 , -3868.4234,    48.579 ]])

    In [126]: np.diff( a[0,:,0] ), np.diff( b[0,:,0] ), np.diff( a[0,:,3] ), np.diff( b[0,:,3] ), np.diff( a[0,:,0] )/np.diff( a[0,:,3] ), np.diff( b[0,:,0] )/np.diff( b[0,:,3] )
    Out[126]: 
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 , -2154.4702]),   A.dx mm
    A([ 2995.0267,     9.9284,   990.094 ,     9.9284,   990.094 , -1918.6317]),   B.dx mm
    A([ 15.3967,       0.0524,   5.0879,       0.0524,   5.0235,  23.5301]),       A.dt ns
    A([ 15.3967,       0.0524,   5.1322,       0.0524,   5.1363,  22.7083]),       B.dt ns
    A([ 194.5238,    189.5833,  194.5969,    189.5833,  197.0937,  -91.5622]),     A.gv mm/ns
    A([ 194.5238,    189.5833,  192.9167,    189.5833,  192.7654,  -84.4902]))     B.gv mm/ns

    ## consistent issue, slow LS and MO groupvel in CFG4 (looking like Ac groupvel)




::

    112 G4VParticleChange* G4VDiscreteProcess::PostStepDoIt(
    113                             const G4Track& ,
    114                             const G4Step&
    115                             )
    116 {
    117 //  clear NumberOfInteractionLengthLeft
    118     ClearNumberOfInteractionLengthLeft();
    119 
    120     return pParticleChange;
    121 }






tconcentric check
--------------------

::

    In [2]: ab.sel = "TO BT BT BT BT SA"    ## straight thru selection

    In [3]: a,b = ab.rpost()

    In [4]: a.shape
    Out[4]: (669843, 6, 4)

    In [5]: b.shape
    Out[5]: (671267, 6, 4)

    In [7]: a[0]    ## positions match, times off a little
    Out[7]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4974],
           [ 3004.9551,     0.    ,     0.    ,    15.5498],
           [ 3995.0491,     0.    ,     0.    ,    20.6377],
           [ 4004.9776,     0.    ,     0.    ,    20.6901],
           [ 4995.0716,     0.    ,     0.    ,    25.7136]])

    In [8]: b[0]
    Out[8]: 
    A()sliced
    A([[    0.    ,     0.    ,     0.    ,     0.1007],
           [ 2995.0267,     0.    ,     0.    ,    15.4934],
           [ 3004.9551,     0.    ,     0.    ,    15.5458],
           [ 3995.0491,     0.    ,     0.    ,    20.682 ],
           [ 4004.9776,     0.    ,     0.    ,    20.7344],
           [ 4995.0716,     0.    ,     0.    ,    25.8666]])


    In [35]: np.diff(a[0,:,0])/np.diff(a[0,:,3])  ## ratio of x diff to t diff -> groupvel in Gd Ac LS Ac MO for  429.5686 nm
    A([ 194.5238,  189.5833,  194.5969,  189.5833,  197.0937])

    In [36]: np.diff(b[0,:,0])/np.diff(b[0,:,3])
    A([ 194.5747,  189.5833,  192.7654,  189.5833,  192.9167])

    In [13]: np.diff(a[0,:,0])/np.diff(a[0,:,3]) - np.diff(b[0,:,0])/np.diff(b[0,:,3])
    A([-0.0509,  0.    ,  1.8315,  0.    ,  4.177 ])    ## mm/ns

    ## fairly close, possibly can attribute to interpolation differences ???




Review
--------

* http://www.hep.man.ac.uk/u/roger/PHYS10302/lecture15.pdf
* http://web.ift.uib.no/AMOS/PHYS261/opticsPDF/Examples_solutions_phys263.pdf

::
                
    .
          c          w  dn           c           
    vg = --- (  1 +  -- ---  )   ~  --- (  1 +   ?  )
          n          n  dw           n              


     d logn      dn   1  
     ------ =   ---  --- 
      dw         dw   n


     d logw      dw   1             dn/n       dn   w
     ------ =   ---  ---    ->     -----  =    ---  -
      dn         dn   w            d logw       dw   n


     c          dn / n 
    --- ( 1 +   ---    )
     n          d logw


     c          dn  
     -   +   c  ---
     n          dlogw




                c         
    vg =  ---------------        # angular freq proportional to E for light     
            n + E dn/dE

    G4 using this energy domain approach approximating the dispersion part E dn/dE as shown below

                c                  n1 - n0         n1 - n0               dn        dn    dE          
    vg =  -----------       ds = ------------  =  ------------     ~   ------  =  ---- ------- =  E dn/dE 
           nn +  ds               log(E1/E0)      log E1 - log E0      d(logE)     dE   dlogE        
  



Now get G4 warnings when run without groupvel option
-------------------------------------------------------

::

    634   accuracy = theVelocityChange/c_light - 1.0;
    635   if (accuracy > accuracyForWarning) {
    636     itsOKforVelocity = false;
    637     nError += 1;
    638     exitWithError = exitWithError ||  (accuracy > accuracyForException);
    639 #ifdef G4VERBOSE
    640     if (nError < maxError) {
    641       G4cout << "  G4ParticleChange::CheckIt    : ";
    642       G4cout << "the velocity is greater than c_light  !!" << G4endl;
    643       G4cout << "  Velocity:  " << theVelocityChange/c_light  <<G4endl;
    644       G4cout << aTrack.GetDefinition()->GetParticleName()
    645          << " E=" << aTrack.GetKineticEnergy()/MeV
    646          << " pos=" << aTrack.GetPosition().x()/m
    647          << ", " << aTrack.GetPosition().y()/m
    648          << ", " << aTrack.GetPosition().z()/m
    649          <<G4endl;
    650     }
    651 #endif
    652   }



    2016-11-10 17:03:42.091 INFO  [373895] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
      G4ParticleChange::CheckIt    : the velocity is greater than c_light  !!
      Velocity:  1.00069
    opticalphoton E=2.88335e-06 pos=1.18776, -0.130221, 2.74632
          -----------------------------------------------
            G4ParticleChange Information  
          -----------------------------------------------
            # of 2ndaries       :                    0
          -----------------------------------------------
            Energy Deposit (MeV):                    0
            Non-ionizing Energy Deposit (MeV):                    0
            Track Status        :                Alive
            True Path Length (mm) :                3e+03
            Stepping Control      :                    0
        First Step In the voulme  : 
        Last Step In the voulme  : 
            Mass (GeV)   :                    0
            Charge (eplus)   :                    0
            MagneticMoment   :                    0
                    :  =                    0*[e hbar]/[2 m]
            Position - x (mm)   :             1.19e+03
            Position - y (mm)   :                 -130
            Position - z (mm)   :             2.75e+03
            Time (ns)           :                 9.98
            Proper Time (ns)    :                    0
            Momentum Direct - x :                0.397
            Momentum Direct - y :              -0.0435
            Momentum Direct - z :                0.917
            Kinetic Energy (MeV):             2.88e-06
            Velocity  (/c):                    1
            Polarization - x    :                0.918
            Polarization - y    :               0.0188
            Polarization - z    :               -0.396
      G4ParticleChange::CheckIt    : the velocity is greater than c_light  !!
      Velocity:  1.00069
    opticalphoton E=2.88335e-06 pos=1.18776, -0.130221, 2.74632
          -----------------------------------------------

::

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







Opticks GROUPVEL
------------------

::

    simon:cfg4 blyth$ opticks-find GROUPVEL 
    ./cfg4/CPropLib.cc: GROUPVEL kludge causing "generational" confusion
    ./cfg4/CPropLib.cc:             LOG(info) << "CPropLib::makeMaterialPropertiesTable applying GROUPVEL kludge" ; 
    ./cfg4/CPropLib.cc:             addProperty(mpt, "GROUPVEL", prop );
    ./cfg4/CPropLib.cc:    bool groupvel = strcmp(lkey, "GROUPVEL") == 0 ; 
    ./cfg4/CTraverser.cc:const char* CTraverser::GROUPVEL = "GROUPVEL" ; 
    ./cfg4/CTraverser.cc:    // First get of GROUPVEL property creates it 
    ./cfg4/CTraverser.cc:            G4MaterialPropertyVector* gv = mpt->GetProperty(GROUPVEL);  
    ./cfg4/tests/CInterpolationTest.cc:    const char* mkeys_1 = "GROUPVEL,,," ;
    ./ggeo/GGeoTestConfig.cc:const char* GGeoTestConfig::GROUPVEL_ = "groupvel"; 
    ./ggeo/GGeoTestConfig.cc:    else if(strcmp(k,GROUPVEL_)==0)   arg = GROUPVEL ; 
    ./ggeo/GGeoTestConfig.cc:        case GROUPVEL       : setGroupvel(s)       ;break;
    ./ggeo/GMaterialLib.cc:"group_velocity:GROUPVEL,"
    ./cfg4/CTraverser.hh:        static const char* GROUPVEL ; 
    ./ggeo/GGeoTestConfig.hh:                      GROUPVEL,
    ./ggeo/GGeoTestConfig.hh:       static const char* GROUPVEL_ ; 
    simon:opticks blyth$ 



G4 GROUPVEL
--------------

::

    simon:geant4_10_02_p01 blyth$ find source -name '*.*' -exec grep -H GROUPVEL {} \;
    source/materials/include/G4MaterialPropertiesTable.hh:// Updated:     2005-05-12 add SetGROUPVEL() by P. Gumplinger
    source/materials/include/G4MaterialPropertiesTable.hh:    G4MaterialPropertyVector* SetGROUPVEL();
    source/materials/include/G4MaterialPropertiesTable.icc:  //2- So we have a data race if two threads access the same element (GROUPVEL)
    source/materials/include/G4MaterialPropertiesTable.icc:  //   at the bottom of the code, one thread in SetGROUPVEL(), and the other here
    source/materials/include/G4MaterialPropertiesTable.icc:  //3- SetGROUPVEL() is protected by a mutex that ensures that only
    source/materials/include/G4MaterialPropertiesTable.icc:  //   the same problematic key (GROUPVEL) the mutex will be used.
    source/materials/include/G4MaterialPropertiesTable.icc:  //5- As soon as a thread acquires the mutex in SetGROUPVEL it checks again
    source/materials/include/G4MaterialPropertiesTable.icc:  //   if the map has GROUPVEL key, if so returns immediately.
    source/materials/include/G4MaterialPropertiesTable.icc:  //   group velocity only once even if two threads enter SetGROUPVEL together
    source/materials/include/G4MaterialPropertiesTable.icc:  if (G4String(key) == "GROUPVEL") return SetGROUPVEL();
    source/materials/src/G4MaterialPropertiesTable.cc:// Updated:     2005-05-12 add SetGROUPVEL(), courtesy of
    source/materials/src/G4MaterialPropertiesTable.cc:G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    source/materials/src/G4MaterialPropertiesTable.cc:  // check if "GROUPVEL" already exists
    source/materials/src/G4MaterialPropertiesTable.cc:  itr = MPT.find("GROUPVEL");
    source/materials/src/G4MaterialPropertiesTable.cc:  // add GROUPVEL vector
    source/materials/src/G4MaterialPropertiesTable.cc:  // fill GROUPVEL vector using RINDEX values
    source/materials/src/G4MaterialPropertiesTable.cc:    G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:      G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:        G4Exception("G4MaterialPropertiesTable::SetGROUPVEL()", "mat205",
    source/materials/src/G4MaterialPropertiesTable.cc:  this->AddProperty( "GROUPVEL", groupvel );
    source/processes/optical/src/G4OpBoundaryProcess.cc:           Material2->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    source/track/src/G4Track.cc:    //  and get new GROUPVELOCITY table if necessary 
    source/track/src/G4Track.cc:      groupvel = mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL");
    source/track/src/G4Track.cc:    // values stored in GROUPVEL material properties vector
    simon:geant4_10_02_p01 blyth$ 




G4Track.cc::

    ///
    ///  GROUPVEL  material property lookup just like RINDEX
    ///            the peculiarity is that the property is 
    ///            derived from RINDEX at first access by special casing in GetProperty
    ///

    317    // cached values for CalculateVelocity  
    318    mutable G4Material*               prev_mat;
    319    mutable G4MaterialPropertyVector* groupvel;
    320    mutable G4double                  prev_velocity;
    321    mutable G4double                  prev_momentum;
    322 


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



/usr/local/opticks/externals/g4/geant4_10_02_p01/source/processes/optical/src/G4OpBoundaryProcess.cc::

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
     ///
     ///     such velocity setting not in DsG4OpBoundaryProcess
     ///
     539 
     540         if ( theStatus == Detection ) InvokeSD(pStep);
     541 
     542         return G4VDiscreteProcess::PostStepDoIt(aTrack, aStep);
     543 }
     544 
     545 void G4OpBoundaryProcess::BoundaryProcessVerbose() const
     546 {




source/materials/include/G4MaterialPropertiesTable.icc::

    115 inline G4MaterialPropertyVector*
    116 G4MaterialPropertiesTable::GetProperty(const char *key)
    117 {
    118   // Returns a Material Property Vector corresponding to a key
    119 
    120   //Important Note for MT. adotti 17 Feb 2016
    121   //In previous implementation the following line was at the bottom of the
    122   //function causing a rare race-condition.
    123   //Moving this line here from the bottom solves the problem because:
    124   //1- Map is accessed only via operator[] (to insert) and find() (to search),
    125   //   and these are thread safe if done on separate elements.
    126   //   See notes on data-races at:
    127   //   http://www.cplusplus.com/reference/map/map/operator%5B%5D/
    128   //   http://www.cplusplus.com/reference/map/map/find/
    129   //2- So we have a data race if two threads access the same element (GROUPVEL)
    130   //   one in read and one in write mode. This was happening with the line
    131   //   at the bottom of the code, one thread in SetGROUPVEL(), and the other here
    132   //3- SetGROUPVEL() is protected by a mutex that ensures that only
    133   //   one thread at the time will execute its code
    134   //4- The if() statement guarantees that only if two threads are searching
    135   //   the same problematic key (GROUPVEL) the mutex will be used.
    136   //   Different keys do not lock (good for performances)
    137   //5- As soon as a thread acquires the mutex in SetGROUPVEL it checks again
    138   //   if the map has GROUPVEL key, if so returns immediately.
    139   //   This "double check" allows to execute the heavy code to calculate
    140   //   group velocity only once even if two threads enter SetGROUPVEL together
    141   if (G4String(key) == "GROUPVEL") return SetGROUPVEL();
    142 
    143   MPTiterator i;
    144   i = MPT.find(G4String(key));
    145   if ( i != MPT.end() ) return i->second;
    146   return NULL;
    147 }

    /// computing a GROUPVEL property vector at first access cause lots of hassle, 
    /// given that RINDEX is constant, should just up front compute GROUPVEL for 
    /// all materials before any event handling happens




::

    119 G4MaterialPropertyVector* G4MaterialPropertiesTable::SetGROUPVEL()
    120 {
    ...
    141   G4MaterialPropertyVector* groupvel = new G4MaterialPropertyVector();
    142 
    146   G4double E0 = rindex->Energy(0);
    147   G4double n0 = (*rindex)[0];
    154   
    160   G4double E1 = rindex->Energy(1);
    161   G4double n1 = (*rindex)[1];
    168 
    169   G4double vg;
    173   vg = c_light/(n0+(n1-n0)/std::log(E1/E0));
    174 
          //   before the loop
          //            E0 = Energy(0)   E1 = Energy(1)      Energy(0) n[0]
          //

    177   if((vg<0) || (vg>c_light/n0))  { vg = c_light/n0; }
    178 
    179   groupvel->InsertValues( E0, vg );
    180 
    184   for (size_t i = 2; i < rindex->GetVectorLength(); i++)
    185   {
    186        vg = c_light/( 0.5*(n0+n1)+(n1-n0)/std::log(E1/E0));

            /// 
            /// note the sleight of hand the same (n1-n0)/std::log(E1/E0) is used for 1st 2 values
            ///

    187 
    190        if((vg<0) || (vg>c_light/(0.5*(n0+n1))))  { vg = c_light/(0.5*(n0+n1)); }

              // at this point in the loop
              //
              // i = 2,    E0 = Energy(0) E1 = Energy(1)    (Energy(0)+Energy(1))/2   // 1st pass using pre-loop settings
              // i = 3,    E0 = Energy(1) E1 = Energy(2)    (Energy(1)+Energy(2))/2   // 2nd pass E0,n0,E1,n1 shunted   
              // i = 4,    E0 = Energy(2) E1 = Energy(3)    (Energy(2)+Energy(3))/2   // 3rd pass E0,n0,E1,n1 shunted   
              //  ....
              // i = N-1   E0 = Energy(N-3)  E1 = Energy(N-2)   (Energy(N-3)+Energy(N-2))/2  


    191        groupvel->InsertValues( 0.5*(E0+E1), vg );
    195        E0 = E1;
    196        n0 = n1;
    197        E1 = rindex->Energy(i);
    198        n1 = (*rindex)[i];
    205   }
    ///
    ///       after the loop 
    ///       "i = N"      E0 = Energy(N-2)   E1 = Energy(N-1)         Energy(N-1)
    ///
    ///     hmmm a difference of bins is needed, but in order not to loose a bin
    ///     a tricky manoever is used of using the 1st and last bin and 
    ///     the average of the body bins
    ///     which means the first bin is half width, and last is 1.5 width
    ///
    ///         0  +  1  +  2  +  3  +  4  +  5        <--- 6 original values
    ///         |    /     /     /     /      |
    ///         |   /     /     /     /       |
    ///         0  1     2     3     4        5        <--- still 6 
    ///
    ///  
    ///
    206 
    209   vg = c_light/(n1+(n1-n0)/std::log(E1/E0));
    213   if((vg<0) || (vg>c_light/n1))  { vg = c_light/n1; }
    214   groupvel->InsertValues( E1, vg );
    ... 
    220   
    221   this->AddProperty( "GROUPVEL", groupvel );
    222   
    223   return groupvel;
    224 }

    ///
    ///           Argh... my domain checking cannot to be working...
    ///           this is sticking values midway in energy 
    ///
    ///           Opticks material texture requires fixed domain raster... 
    ///           so either interpolate to get that or adjust the calc ???
    ///


::

   ml = np.load("GMaterialLib.npy")
   wl = np.linspace(60,820,39)
   ri = ml[0,0,:,0]

   c_light = 299.792

   w0 = wl[:-1]
   w1 = wl[1:]

   n0 = ri[:-1]
   n1 = ri[1:]

    In [41]: c_light/(n0 + (n1-n0)/np.log(w1/w0))    # douple flip for e to w, one for reciprocal, one for order ???
    Out[41]: 
    array([ 206.2411,  206.2411,  206.2411,  106.2719,  114.2525, -652.0324,  125.2658,  210.3417,  215.9234,  221.809 ,  228.0242,  234.5973,  207.5104,  209.0361,  210.5849,  212.1565,  213.7514,
            207.991 ,  206.1923,  205.4333,  205.883 ,  206.8385,  207.5627,  208.0809,  206.0739,  205.295 ,  205.4116,  205.5404,  205.7735,  206.0065,  206.2412,  205.3909,  204.2895,  204.3864,
            204.4841,  204.5806,  204.6679,  202.8225])










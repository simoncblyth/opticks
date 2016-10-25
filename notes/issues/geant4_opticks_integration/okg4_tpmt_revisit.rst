OKG4 tpmt revisited
======================

Perhaps surface conversion not yet implemented for CTestDetector ?
---------------------------------------------------------------------

::

    097 // called by CGeometry::init for CGDMLDetector case
     98 void CSurLib::convert(CDetector* detector)
     99 {
    100     setDetector(detector);
    101     unsigned numSur = m_surlib->getNumSur();
    102     LOG(info) << "CSurLib::convert  numSur " << numSur  ;
    103     for(unsigned i=0 ; i < numSur ; i++)
    104     {
    105         GSur* sur = m_surlib->getSur(i);
    106         G4OpticalSurface* os = makeOpticalSurface(sur);


G4 EFFICIENCY
--------------

Need cathode optical surface with EFFICIENCY, where did it go ?

::

    simon:source blyth$ find . -name '*.cc' -exec grep -H EFFICIENCY {} \;
    ./global/HEPNumerics/src/G4ConvergenceTester.cc:   out << std::setw(20) << "EFFICIENCY = " << std::setw(13)  << efficiency << G4endl;
    ./processes/optical/src/G4OpBoundaryProcess.cc:              aMaterialPropertiesTable->GetProperty("EFFICIENCY");
    simon:source blyth$ find . -name '*.hh' -exec grep -H EFFICIENCY {} \;
    simon:source blyth$ 

    387               PropertyPointer =
    388               aMaterialPropertiesTable->GetProperty("EFFICIENCY");
    389               if (PropertyPointer) {
    390                       theEfficiency =
    391                       PropertyPointer->Value(thePhotonMomentum);
    392               }


    306 inline
    307 void G4OpBoundaryProcess::DoAbsorption()
    308 {
    309               theStatus = Absorption;
    310 
    311               if ( G4BooleanRand(theEfficiency) ) {
    312 
    313                  // EnergyDeposited =/= 0 means: photon has been detected
    314                  theStatus = Detection;
    315                  aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    316               }
    317               else {
    318                  aParticleChange.ProposeLocalEnergyDeposit(0.0);
    319               }
    320 
    321               NewMomentum = OldMomentum;
    322               NewPolarization = OldPolarization;
    323 
    324 //              aParticleChange.ProposeEnergy(0.0);
    325               aParticleChange.ProposeTrackStatus(fStopAndKill);
    326 }

* caution are actually used the custom cfg4/DsG4OpBoundaryProcess.cc



Divergence following many changes : g4 not stopping at cathode ?
----------------------------------------------------------------------------

tpmt was formerly in near perfect agreement, recent changes have caused divergence


CG4 now needs proper surface handling, not a kludge::

    207 unsigned int OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst)
    208 {
    209     G4StepStatus status = point->GetStepStatus()  ;
    210     // TODO: cache the relevant process objects, so can just compare pointers ?
    211     const G4VProcess* process = point->GetProcessDefinedStep() ;
    212     const G4String& processName = process ? process->GetProcessName() : "NoProc" ;
    213 
    214     bool transportation = strcmp(processName,"Transportation") == 0 ;
    215     bool scatter = strcmp(processName, "OpRayleigh") == 0 ;
    216     bool absorption = strcmp(processName, "OpAbsorption") == 0 ;
    217 
    218     unsigned int flag(0);
    219     if(absorption && status == fPostStepDoItProc )
    220     {
    221         flag = BULK_ABSORB ;
    222     }
    223     else if(scatter && status == fPostStepDoItProc )
    224     {
    225         flag = BULK_SCATTER ;
    226     }
    227     else if(transportation && status == fWorldBoundary )
    228     {
    229         flag = SURFACE_ABSORB ;   // kludge for fWorldBoundary - no surface handling yet 
    230     }
    231     else if(transportation && status == fGeomBoundary )
    232     {
    233         flag = OpBoundaryFlag(bst) ; // BOUNDARY_TRANSMIT/BOUNDARY_REFLECT/NAN_ABORT/SURFACE_ABSORB/SURFACE_DETECT
    ///
    //         ^^^^^^^^ SD from here ? perhaps missing optical surface with EFFICIENCY ?
    ///
    234     }
    235     return flag ;
    236 }









::

    tpmt-t () 
    { 
        tpmt-;
        tpmt-- --okg4 --compute
    }


    [2016-10-25 13:13:45,923] p40907 {/Users/blyth/opticks/ana/tpmt.py:146} INFO -  a : PmtInBox/torch/ 10 :  20161025-1313 /tmp/blyth/opticks/evt/PmtInBox/torch/10/fdom.npy 
    [2016-10-25 13:13:45,923] p40907 {/Users/blyth/opticks/ana/tpmt.py:147} INFO -  b : PmtInBox/torch/-10 :  20161025-1313 /tmp/blyth/opticks/evt/PmtInBox/torch/-10/fdom.npy 


       A:seqhis_ana   10:PmtInBox 
                 8cd        0.676           6762       [3 ] TO BT SA
                 7cd        0.221           2209       [3 ] TO BT SD
                8ccd        0.047            472       [4 ] TO BT BT SA
                  4d        0.038            384       [2 ] TO AB
                 86d        0.006             57       [3 ] TO SC SA
                 4cd        0.004             41       [3 ] TO BT AB
                 8bd        0.003             26       [3 ] TO BR SA
                4ccd        0.002             24       [4 ] TO BT BT AB
               86ccd        0.001              6       [5 ] TO BT BT SC SA
                8c6d        0.001              6       [4 ] TO SC BT SA
                 46d        0.000              3       [3 ] TO SC AB
                7c6d        0.000              3       [4 ] TO SC BT SD
          8ccccc6ccd        0.000              1       [10] TO BT BT SC BT BT BT BT BT SA
                866d        0.000              1       [4 ] TO SC SC SA
                86bd        0.000              1       [4 ] TO BR SC SA
               8c66d        0.000              1       [5 ] TO SC SC BT SA
                 4bd        0.000              1       [3 ] TO BR AB
              8cbbcd        0.000              1       [6 ] TO BT BR BR BT SA
          ccbccc6ccd        0.000              1       [10] TO BT BT SC BT BT BT BR BT BT
                           10000         1.00 
       B:seqhis_ana   -10:PmtInBox 
            8ccccccd        0.431           4315       [8 ] TO BT BT BT BT BT BT SA
             8ccbccd        0.285           2850       [7 ] TO BT BT BR BT BT SA
                8ccd        0.045            455       [4 ] TO BT BT SA
                  4d        0.039            386       [2 ] TO AB
          ccccbccccd        0.033            334       [10] TO BT BT BT BT BR BT BT BT BT
            4ccccccd        0.031            314       [8 ] TO BT BT BT BT BT BT AB
             4ccbccd        0.021            207       [7 ] TO BT BT BR BT BT AB
           8cccccccd        0.017            168       [9 ] TO BT BT BT BT BT BT BT SA
           8cccbcccd        0.015            154       [9 ] TO BT BT BT BR BT BT BT SA
          8ccbcccccd        0.014            144       [10] TO BT BT BT BT BT BR BT BT SA
           86ccccccd        0.006             60       [9 ] TO BT BT BT BT BT BT SC SA
                 86d        0.005             55       [3 ] TO SC SA
          cccbcbcccd        0.004             40       [10] TO BT BT BT BR BT BR BT BT BT
          bccbcbcccd        0.004             40       [10] TO BT BT BT BR BT BR BT BT BR
                 4cd        0.004             37       [3 ] TO BT AB
                4ccd        0.004             36       [4 ] TO BT BT AB
          cccbcccccd        0.003             33       [10] TO BT BT BT BT BT BR BT BT BT
          8cccbbcccd        0.003             32       [10] TO BT BT BT BR BR BT BT BT SA
            86ccbccd        0.003             31       [8 ] TO BT BT BR BT BT SC SA
          cccccccccd        0.003             29       [10] TO BT BT BT BT BT BT BT BT BT
                           10000         1.00 
       A:seqmat_ana   10:PmtInBox 
                 5e4        0.897           8971       [3 ] MO Py Bk
                c4e4        0.047            472       [4 ] MO Py MO Rk
                  44        0.038            384       [2 ] MO MO
                 c44        0.008             83       [3 ] MO MO Rk
                 ee4        0.004             41       [3 ] MO Py Py
                44e4        0.002             24       [4 ] MO Py MO MO
                5e44        0.001              9       [4 ] MO MO Py Bk
               c44e4        0.001              6       [5 ] MO Py MO MO Rk
                 444        0.000              4       [3 ] MO MO MO
                c444        0.000              2       [4 ] MO MO MO Rk
          eedede44e4        0.000              1       [10] MO Py MO MO Py Vm Py Vm Py Py
          c4edbe44e4        0.000              1       [10] MO Py MO MO Py OV Vm Py MO Rk
              c4eee4        0.000              1       [6 ] MO Py Py Py MO Rk
               5e444        0.000              1       [5 ] MO MO MO Py Bk
                           10000         1.00 
       B:seqmat_ana   -10:PmtInBox 
             4ebd5e4        0.400           4004       [7 ] MO Py Bk Vm OV Py MO
              4e55e4        0.285           2850       [6 ] MO Py Bk Bk Py MO
                  44        0.047            466       [2 ] MO MO
                 4e4        0.045            455       [3 ] MO Py MO
            44ebd5e4        0.035            349       [8 ] MO Py Bk Vm OV Py MO MO
             4e5d5e4        0.031            311       [7 ] MO Py Bk Vm Bk Py MO
             44e55e4        0.024            238       [7 ] MO Py Bk Bk Py MO MO
            4edbd5e4        0.017            168       [8 ] MO Py Bk Vm OV Vm Py MO
           4eddbd5e4        0.014            144       [9 ] MO Py Bk Vm OV Vm Vm Py MO
          edbdbbd5e4        0.014            136       [10] MO Py Bk Vm OV OV Vm OV Vm Py
            4ebdd5e4        0.013            129       [8 ] MO Py Bk Vm Vm OV Py MO
          bdbdbbd5e4        0.009             85       [10] MO Py Bk Vm OV OV Vm OV Vm OV
          4ebdbbd5e4        0.007             66       [10] MO Py Bk Vm OV OV Vm OV Py MO
          ebdbbdd5e4        0.004             40       [10] MO Py Bk Vm Vm OV OV Vm OV Py
          bbdbbdd5e4        0.004             40       [10] MO Py Bk Vm Vm OV OV Vm OV OV
                 ee4        0.004             37       [3 ] MO Py Py
                44e4        0.004             37       [4 ] MO Py MO MO
          4ebddbd5e4        0.003             32       [10] MO Py Bk Vm OV Vm Vm OV Py MO
          4edbdbd5e4        0.003             29       [10] MO Py Bk Vm OV Vm OV Vm Py MO
           4ebddd5e4        0.003             28       [9 ] MO Py Bk Vm Vm Vm OV Py MO
                           10000         1.00 







FIXED : tpmt takes exception to duplicated OpaqueVacuum material name
------------------------------------------------------------------------

::

    tpmt-- --okg4 --compute


    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@26] CMaterialBridge::initMap nmat (G4Material::GetNumberOfMaterials) 6
    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@45]  i   0 name                          MineralOil shortname                          MineralOil index     3
    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@45]  i   1 name                               Pyrex shortname                               Pyrex index    13
    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@45]  i   2 name                              Vacuum shortname                              Vacuum index    12
    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@45]  i   3 name                            Bialkali shortname                            Bialkali index     4
    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@45]  i   4 name                        OpaqueVacuum shortname                        OpaqueVacuum index    10
    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@45]  i   5 name                        OpaqueVacuum shortname                        OpaqueVacuum index    10
    2016-10-25 12:28:33.907 INFO  [3373620] [CMaterialBridge::initMap@52]  nmat 6 m_g4toix.size() 6 m_ixtoname.size() 5
    Assertion failed: (m_ixtoname.size() == nmat && "there is probably a duplicated material name"), function initMap, file /Users/blyth/opticks/cfg4/CMaterialBridge.cc, line 60.
    Process 38758 stopped

Fixed by G4Material recycling in CPropLib.





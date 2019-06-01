ckm-okg4-initial-comparisons
================================

Context :doc:`ckm-okg4-initial-comparisons-reveal-idom-bug`

Issues
----------

* :doc:`ckm-okg4-initial-comparisons-sensor-matching-yet-again`

  * G4 missing SD/SA : resulting in history divergence 
  * Seen this kinda problem before several times before : sensitivity fails to travel OR be translated  


ckm-- source events
-----------------------

::

    ckm-- () 
    { 
        g4-;
        g4-export;
        CerenkovMinimal
    }



"source" events from uninstrumented on G4 side ckm-- CerenkovMinimal executable, paths relative to geocache dir::

    [blyth@localhost 1]$ np.py source/evt/g4live/natural/{-1,1} -T
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/-1
    . :                          source/evt/g4live/natural/-1/ht.npy :          (108, 4, 4) : f151301a12d1874e9447fd916e7f8719 : 20190530-2247 
    . :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190530-2247 
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/source/evt/g4live/natural/1
    . :                         source/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190530-2247 
    . :                         source/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190530-2247 
    . :                           source/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190530-2247 
    . :                           source/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190530-2247 



ckm-okg4 : From gensteps bi-simulation comparison (-1:G4, 1:OK)
--------------------------------------------------------------------

::

    ckm-okg4 () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural
    }


OKG4Test events fully instrumented on both sides using the CFG4 CRecorder machinery::

    [blyth@localhost 1]$ np.py OKG4Test/evt/g4live/natural/{-1,1} -T
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/OKG4Test/evt/g4live/natural/-1
    . :                      OKG4Test/evt/g4live/natural/-1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ht.npy :            (0, 4, 4) : 40d1000a029fc713333b79245d7141c1 : 20190531-1448 
    . :                      OKG4Test/evt/g4live/natural/-1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ox.npy :          (221, 4, 4) : 0c933fd9fdab9d2975af9e6871351e46 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ph.npy :          (221, 1, 2) : 0a50e4992b98714e0391cd6d8deadc9e : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/ps.npy :          (221, 1, 4) : 2f17ee76054cc1040f30bee0a8a0153e : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/rs.npy :      (221, 10, 1, 4) : 629500c344dc05dbc6777ccf6f386fe5 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/rx.npy :      (221, 10, 2, 4) : 2ce8d2aafab81f6d6f0e6a1cc1877646 : 20190531-1448 
    . :                        OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190531-1448 
    /home/blyth/local/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/OKG4Test/evt/g4live/natural/1
    . :                       OKG4Test/evt/g4live/natural/1/fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/gs.npy :            (1, 6, 4) : f21b4f0138c122a64319243596bb2228 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ht.npy :           (42, 4, 4) : a5dcecca8a61f6ef3e324edac8f36361 : 20190531-1448 
    . :                       OKG4Test/evt/g4live/natural/1/idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ox.npy :          (221, 4, 4) : bbf07d0da33758272b447ba44655decd : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ph.npy :          (221, 1, 2) : c60500d6d56b530b1c55bf6b14c34a15 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/ps.npy :          (221, 1, 4) : ad7498e182d8df1bf720c0ba0e72736c : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/rs.npy :      (221, 10, 1, 4) : ce59ba752de205fb16062260c6383503 : 20190531-1448 
    . :                         OKG4Test/evt/g4live/natural/1/rx.npy :      (221, 10, 2, 4) : c085570c57f4749d13475312fcd16fb5 : 20190531-1448 


Observations:

* OK: misses so.npy source photons 

  * TODO: add collection of just generated source photons, behind a WITH_SOURCE_PHOTONS macro, 
    this will be more convenient that having to do a separate bouncemax zero run to check the source photons

* OK: array digests match between 1st executable CerenkovMinimal and 2nd OKG4Test : as is expected because same input gensteps and same code

* G4 : so.npy source photons between 1st and 2nd are not an exact match, but they are very close::

    [blyth@localhost 1]$ np.py {source,OKG4Test}/evt/g4live/natural/-1/so.npy 
    a :                          source/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : d1b4242225f7ffc7f0ad38a9669562a4 : 20190530-2247 
    b :                        OKG4Test/evt/g4live/natural/-1/so.npy :          (221, 4, 4) : 882f44b7864bfcde55fe2ebe922895e5 : 20190531-1448 
     max(a-b)   5.96e-08  min(a-b)  -5.96e-08 


* G4 : ht.npy getting zero hits in 2nd executable 



ckm-ip : after evt.py nload.py pfx generalizations to find OKG4Test events
------------------------------------------------------------------------------

::

    ckm-ip () 
    { 
        ipython --pdb $(which ckm.py) -i $*
    }


Compare history frequency::

    In [1]: a.seqhis_ana.table
    Out[1]: 
    .                seqhis_ana  1:g4live:OKG4Test 
    .                                221         1.00 
    0000              3c1        0.638         141      [3 ] CK BT MI
    0001               71        0.190          42      [2 ] CK SD
    0002               81        0.167          37      [2 ] CK SA
    0003             3cc1        0.005           1      [4 ] CK BT BT MI
    .                                221         1.00 

    In [2]: b.seqhis_ana.table
    Out[2]: 
    .                seqhis_ana  -1:g4live:OKG4Test 
    .                                221         1.00 
    0000              3c1        0.606         134      [3 ] CK BT MI
    0001            3ccc1        0.326          72      [5 ] CK BT BT BT MI
    0002             3cb1        0.023           5      [4 ] CK BR BT MI
    0003           3ccbc1        0.014           3      [6 ] CK BT BR BT BT MI
    0004       bbccbbbbb1        0.009           2      [10] CK BR BR BR BR BR BT BT BR BR
    .                                221         1.00 


* G4 missing SD/SA 


Compare histories of first 20 photons::

    In [6]: a.seqhis_ls[0:20]
    Out[6]: 
    CK SA
    CK SD
    CK BT MI
    CK SA
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK SD
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK SD

    In [7]: b.seqhis_ls[0:20]
    Out[7]: 
    CK BT BT BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT BT BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT MI
    CK BT BT BT MI


Recorded positions of first few photons, show they are starting out together but history diverges at the SD/SA which 
happens for 1(OK) but not -1(G4).::

    In [8]: a.rposti(0)
    Out[8]: 
    A()sliced
    A([[  0.061,   0.   ,   0.   ,   0.   ],
       [127.659, -35.981,  89.999,   0.726]])

    In [9]: b.rposti(0)
    Out[9]: 
    A()sliced
    A([[   0.061,    0.   ,    0.   ,    0.   ],
       [ 127.659,  -35.981,   89.999,    0.726],   ### history diverges here, OK ends on an SD, G4 continues to BT on   
       [ 149.876,  -42.268,  109.989,    0.879],
       [ 500.015, -140.996,  356.944,    2.875],
       [ 977.783, -398.114, 1000.   ,    5.683]])

    In [10]: a.rposti(2)
    Out[10]: 
    A()sliced
    A([[   0.336,   -0.061,    0.   ,    0.   ],
       [ 500.015,  206.915, -149.327,    2.576],
       [1000.   ,  521.134, -376.019,    4.682]])

    In [11]: b.rposti(2)
    Out[11]: 
    A()sliced
    A([[   0.336,   -0.061,    0.   ,    0.   ],
       [ 500.015,  206.915, -149.327,    2.576],
       [1000.   ,  521.104, -376.019,    4.682]])




Dumping record_id 0 using --dbgseqhis 0x3ccc1
----------------------------------------------

::

    ckm-okg4 --dbgseqhis 0x3ccc1

    ckm-okg4 () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural --args $*
    }

    ...

    2019-06-01 14:02:10.530 INFO  [428380] [CRec::dump@162] CDebug::dump record_id 0  origin[ 0.054-0.011-0.002]   Ori[ 0.054-0.011-0.002] 
    2019-06-01 14:02:10.530 INFO  [428380] [CRec::dump@168]  nstp 4
    ( 0)  CK/BT     FrT                       PRE_SAVE POST_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 1 pid 0 nm 79.0277 mm  ori[    0.054  -0.011  -0.002]  pos[  977.731-398.0881000.002]  )
      pre              Obj0x1899da0           Water          noProc           Undefined pos[      0.000     0.000     0.000]  dir[    0.796  -0.225   0.562]  pol[   -0.571   0.027   0.820]  ns  0.000 nm 79.028 mm/ns 220.306
     post              Det0x189c420           Glass  Transportation        GeomBoundary pos[    127.607   -35.985    90.002]  dir[    0.727  -0.205   0.655]  pol[   -0.660   0.053   0.749]  ns  0.728 nm 79.028 mm/ns 201.203
     )
    ( 1)  BT/BT     FrT                                           POST_SAVE 
    [   1](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 1 pid 0 nm 79.0277 mm  ori[    0.054  -0.011  -0.002]  pos[  977.731-398.0881000.002]  )
      pre              Det0x189c420           Glass  Transportation        GeomBoundary pos[    127.607   -35.985    90.002]  dir[    0.727  -0.205   0.655]  pol[   -0.660   0.053   0.749]  ns  0.728 nm 79.028 mm/ns 201.203
     post              Obj0x1899da0           Water  Transportation        GeomBoundary pos[    149.814   -42.247   110.002]  dir[    0.796  -0.225   0.562]  pol[   -0.571   0.029   0.821]  ns  0.879 nm 79.028 mm/ns 220.306
     )
    ( 2)  BT/BT     FrT                                           POST_SAVE 
    [   2](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 1 pid 0 nm 79.0277 mm  ori[    0.054  -0.011  -0.002]  pos[  977.731-398.0881000.002]  )
      pre              Obj0x1899da0           Water  Transportation        GeomBoundary pos[    149.814   -42.247   110.002]  dir[    0.796  -0.225   0.562]  pol[   -0.571   0.029   0.821]  ns  0.879 nm 79.028 mm/ns 220.306
     post         World0x188d190_PV             Air  Transportation        GeomBoundary pos[    499.946  -140.984   356.954]  dir[    0.568  -0.306   0.764]  pol[    0.781   0.494  -0.383]  ns  2.875 nm 79.028 mm/ns 299.792
     )
    ( 3)  BT/MI     FrT                       POST_SAVE POST_DONE LAST_POST 
    [   3](Stp ;opticalphoton stepNum    4(tk ;opticalphoton tid 1 pid 0 nm 79.0277 mm  ori[    0.054  -0.011  -0.002]  pos[  977.731-398.0881000.002]  )
      pre         World0x188d190_PV             Air  Transportation        GeomBoundary pos[    499.946  -140.984   356.954]  dir[    0.568  -0.306   0.764]  pol[    0.781   0.494  -0.383]  ns  2.875 nm 79.028 mm/ns 299.792
     post                                noMaterial  Transportation       WorldBoundary pos[    977.731  -398.088  1000.002]  dir[    0.568  -0.306   0.764]  pol[    0.781   0.494  -0.383]  ns  5.682 nm 79.028 mm/ns 299.792
     )
    2019-06-01 14:02:10.530 INFO  [428380] [CRec::dump@172]  npoi 0
    2019-06-01 14:02:10.531 INFO  [428380] [CDebug::dump_brief@176] CRecorder::dump_brief m_ctx._record_id        0 m_photon._badflag     0 --dbgseqhis  sas: POST_SAVE POST_DONE LAST_POST 
    2019-06-01 14:02:10.531 INFO  [428380] [CDebug::dump_brief@185]  seqhis            3ccc1    CK BT BT BT MI                                  
    2019-06-01 14:02:10.531 INFO  [428380] [CDebug::dump_brief@190]  mskhis              805    CK|MI|BT
    2019-06-01 14:02:10.531 INFO  [428380] [CDebug::dump_brief@195]  seqmat            11232    Water Glass Water Air Air - - - - - - - - - - - 
    2019-06-01 14:02:10.531 INFO  [428380] [CDebug::dump_sequence@203] CDebug::dump_sequence



What would it take to raise an SA/SD from Geant4 ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* need boundary status Ds::Absorption or Ds::Detection

* examining DsG4OpBoundaryProcess.cc (gdb next "n" stepping) 
  looks like no way 


::

    201 #ifdef USE_CUSTOM_BOUNDARY
    202 unsigned int OpStatus::OpBoundaryFlag(const Ds::DsG4OpBoundaryProcessStatus status)
    203 {
    204     unsigned flag = 0 ;
    205     switch(status)
    206     {
    207         case Ds::FresnelRefraction:
    208         case Ds::SameMaterial:
    209                                flag=BOUNDARY_TRANSMIT;
    210                                break;
    211         case Ds::TotalInternalReflection:
    212         case Ds::FresnelReflection:
    213                                flag=BOUNDARY_REFLECT;
    214                                break;
    215         case Ds::StepTooSmall:
    216                                flag=NAN_ABORT;
    217                                break;
    218         case Ds::Absorption:
    219                                flag=SURFACE_ABSORB ;
    220                                break;
    221         case Ds::Detection:
    222                                flag=SURFACE_DETECT ;
    223                                break;
    224         case Ds::SpikeReflection:
    225                                flag=SURFACE_SREFLECT ;
    226                                break;
    227         case Ds::LobeReflection:
    228         case Ds::LambertianReflection:
    229                                flag=SURFACE_DREFLECT ;
    230                                break;
    231         case Ds::Undefined:
    232         case Ds::BackScattering:
    233         case Ds::NotAtBoundary:
    234         case Ds::NoRINDEX:
    235                       flag=0;
    236                       break;
    237     }
    238     return flag ;
    239 }




::

     ckm-okg4-dbg --dbgseqhis 0x3ccc1
    
    (gdb) b "DsG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)"





generate.cu  only gives SD/SA with associated optical surface at the boundary
---------------------------------------------------------------------------------

::

    548 
    549         slot++ ;
    550 
    551         command = propagate_to_boundary( p, s, rng );
    552         if(command == BREAK)    break ;           // BULK_ABSORB
    553         if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
    554         // PASS : survivors will go on to pick up one of the below flags, 
    555 
    556 
    557         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    558         {
    559             command = propagate_at_surface(p, s, rng);
    560             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    561             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    562         }
    563         else
    564         {
    565             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    566             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    567             // tacit CONTINUE
    568         }
    569 
    570     }   // bounce < bounce_max




DsG4OpBoundaryProcess::DoAbsorption raises Ds::Detection depending on efficiency and random throw
---------------------------------------------------------------------------------------------------





gdb aint lldb
------------------

* seems it can find breakpoints better without arguments in the signature


::

    (gdb) b "DsG4OpBoundaryProcess::DsG4OpBoundaryProcess"
    Breakpoint 3 at 0x7fffefd59087: file /home/blyth/opticks/cfg4/DsG4OpBoundaryProcess.cc, line 149.
    (gdb) b
    No default breakpoint address now.
    (gdb) info breakpoints
    Num     Type           Disp Enb Address            What
    1       breakpoint     keep y   <PENDING>          "DsG4OpBoundaryProcess::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)"
    2       breakpoint     keep y   <PENDING>          DsG4OpBoundaryProcess.cc +245
    3       breakpoint     keep y   0x00007fffefd59087 in DsG4OpBoundaryProcess::DsG4OpBoundaryProcess(CG4*, G4String const&, G4ProcessType) at /home/blyth/opticks/cfg4/DsG4OpBoundaryProcess.cc:149
    (gdb) b "DsG4OpBoundaryProcess::PostStepDoIt"
    Breakpoint 4 at 0x7fffefd595d1: file /home/blyth/opticks/cfg4/DsG4OpBoundaryProcess.cc, line 199.
    (gdb) 



covered this ground before
-------------------------------

::

    [blyth@localhost issues]$ grep -l " SD " *.rst */*.rst
    ckm-okg4-initial-comparisons-sensor-matching-yet-again.rst
    direct_route_needs_AssimpGGeo_convertSensors_equivalent.rst
    g4ok_direct_conversion_of_sensors_review.rst
    g4ok_investigate_zero_hits.rst
    G4OK.rst
    G4OK_SD_Matching.rst
    odd_photon_flag_history.rst
    OKG4Test_no_G4_hits.rst
    OKG4Test_no_OK_hits_again.rst
    OKTest-compute-save.rst
    pmt_distrib.rst
    pmttest.rst
    stratification.rst
    geant4_opticks_integration/broken_pmttest.rst
    geant4_opticks_integration/gui_photon_flag_names_null.rst
    geant4_opticks_integration/missing_cfg4_surface_detect.rst
    geant4_opticks_integration/okg4_tpmt_revisit.rst
    geant4_opticks_integration/optical_step_collection.rst
    groupvel/generational.rst
    [blyth@localhost issues]$ 



* :doc:`geant4_opticks_integration/missing_cfg4_surface_detect`

   Concluded : CGDMLDetector missing Optical Surfaces whereas the CTestDetector has them 

* :doc:`geant4_opticks_integration/okg4_tpmt_revisit`

   tpmt- seqhis_ana 10:PmtInBox statistical comparison with "TO BT SA" and "TO BT SD" matched, 

   Notable quotes:
 
   1. FIXED : CTestDetector::kludgePhotoCathode was incorrectly using dielectric_dielectric
   2. Need cathode optical surface with EFFICIENC, where did it go ?Y


* :doc:`OKG4Test_no_G4_hits`

  lv2sd via cache development while testing with CerenkovMinimal

* :doc:`OKG4Test_no_OK_hits_again`

   X4PhysicalVolume::addBoundary critical for translatin G4 surface into Opticks boundary

* :doc:`ab-blib`

   X4PhysicalVolume::findSurface is attempting to mimic G4OpBoundaryProcess 


* :doc:`g4ok_direct_conversion_of_sensors_review`

   // X4PhysicalVolume::init 
   convertSensors();  // before closeSurfaces as may add some SensorSurfaces"

   GGeoSensor::AddSensorSurfaces
        springs into life GGeo GSkinSurface, GOpticalSurface with the 
        properties of the cathode material (esp EFFICIENCY)





::

     278 // invoked pre-cache by GGeo::add(GMaterial* material) AssimpGGeo::convertMaterials
     279 void GMaterialLib::add(GMaterial* mat)
     280 {
     281     if(mat->hasProperty("EFFICIENCY"))
     282     {
     283         LOG(LEVEL) << " MATERIAL WITH EFFICIENCY " ;
     284         setCathode(mat) ;
     285     }
     286 
     287     bool with_lowercase_efficiency = mat->hasProperty("efficiency") ;
     288     assert( !with_lowercase_efficiency );
     289 
     290     assert(!isClosed());
     291     m_materials.push_back(createStandardMaterial(mat));
     292 }


OKG4Test_direct_two_executable_shakedown
===========================================

simstream G4 vs G4 comparison with the two executables : ckm-- and ckm-okg4
----------------------------------------------------------------------------

Implemented SBacktrack::CallSite and using it from CMixMaxRng flat shim 
to see who is calling flat, write those out to simstream files 
with the random number. 

vimdiff between them makes it very apparent that the only way of getting alignment is
to be using exactly the same code for the physics. 

Previously starting from the photons is much easier, as there is so much less 
physics to worry about.

* this observation from seeing the G4-G4 simstreams makes me realize need to switch 
  back to genstep strategy :doc:`strategy_for_Cerenkov_Scintillation_alignment`


::

    00 :   0.519572 :       + 661 G4VEmProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)
     1 :   0.887343 :       + 935 G4VEnergyLossProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)
     2 :   0.469907 :       + 935 G4VEnergyLossProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)
     3 :   0.222093 :        + 22 CLHEP::RandGaussQ::shoot(CLHEP::HepRandomEngine*)
     4 :   0.859488 :      + 2654 G4UrbanMscModel::SampleCosineTheta(double, double)
     5 :   0.546826 :      + 2706 G4UrbanMscModel::SampleCosineTheta(double, double)
     6 :   0.928684 :      + 2763 G4UrbanMscModel::SampleCosineTheta(double, double)
     7 :   0.439334 :       + 554 G4UrbanMscModel::SampleScattering(CLHEP::Hep3Vector const&, double)
     8 :   0.739872 :       + 110 G4UrbanMscModel::SampleDisplacement(double, double)
     9 :   0.608161 :       + 816 G4UrbanMscModel::SampleDisplacement(double, double)
    10 :   0.434261 :        + 74 G4Poisson(double)
    11 :   0.505910 :       + 168 G4UniversalFluctuation::AddExcitation(CLHEP::HepRandomEngine*, double, double, double&, double&, double&)
    12 :   0.092601 :        + 22 CLHEP::RandGaussQ::shoot(CLHEP::HepRandomEngine*)
    13 :   0.417196 :        + 74 G4Poisson(double)
    14 :   0.917940 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    15 :   0.310028 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    16 :   0.392645 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    17 :   0.864323 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    18 :   0.733161 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    19 :   0.377088 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    20 :   0.056634 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    21 :   0.708644 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    22 :   0.290254 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    23 :   0.663173 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    24 :   0.338007 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    25 :   0.026170 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    26 :   0.900784 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    27 :   0.422549 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    28 :   0.778952 :        + 22 CLHEP::RandGaussQ::shoot(CLHEP::HepRandomEngine*)
    29 :   0.956925 :        + 74 G4Poisson(double)
    30 :   0.406647 :      + 2433 C4Cerenkov1042::PostStepDoIt(G4Track const&, G4Step const&)
    31 :   0.490262 :      + 2654 C4Cerenkov1042::PostStepDoIt(G4Track const&, G4Step const&)
    32 :   0.671936 :      + 2749 C4Cerenkov1042::PostStepDoIt(G4Track const&, G4Step const&)


::

    00 :   0.519572 :       + 661 G4VEmProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)
     1 :   0.887343 :       + 935 G4VEnergyLossProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)
     2 :   0.469907 :       + 935 G4VEnergyLossProcess::PostStepGetPhysicalInteractionLength(G4Track const&, double, G4ForceCondition*)
     3 :   0.222093 :        + 22 CLHEP::RandGaussQ::shoot(CLHEP::HepRandomEngine*)
     4 :   0.859488 :      + 2654 G4UrbanMscModel::SampleCosineTheta(double, double)
     5 :   0.546826 :      + 2706 G4UrbanMscModel::SampleCosineTheta(double, double)
     6 :   0.928684 :      + 2763 G4UrbanMscModel::SampleCosineTheta(double, double)
     7 :   0.439334 :       + 554 G4UrbanMscModel::SampleScattering(CLHEP::Hep3Vector const&, double)
     8 :   0.739872 :       + 110 G4UrbanMscModel::SampleDisplacement(double, double)
     9 :   0.608161 :       + 816 G4UrbanMscModel::SampleDisplacement(double, double)
    10 :   0.434261 :        + 74 G4Poisson(double)
    11 :   0.505910 :       + 168 G4UniversalFluctuation::AddExcitation(CLHEP::HepRandomEngine*, double, double, double&, double&, double&)
    12 :   0.092601 :        + 22 CLHEP::RandGaussQ::shoot(CLHEP::HepRandomEngine*)
    13 :   0.417196 :        + 74 G4Poisson(double)
    14 :   0.917940 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    15 :   0.310028 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    16 :   0.392645 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    17 :   0.864323 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    18 :   0.733161 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    19 :   0.377088 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    20 :   0.056634 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    21 :   0.708644 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    22 :   0.290254 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    23 :   0.663173 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    24 :   0.338007 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    25 :   0.026170 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    26 :   0.900784 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    27 :   0.422549 :      + 3421 G4UniversalFluctuation::SampleFluctuations(G4MaterialCutsCouple const*, G4DynamicParticle const*, double, double, double)
    28 :   0.778952 :        + 22 CLHEP::RandGaussQ::shoot(CLHEP::HepRandomEngine*)
    29 :   0.956925 :        + 74 G4Poisson(double)
    30 :   0.406647 :      + 2662 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    31 :   0.490262 :      + 2883 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    32 :   0.671936 :      + 2978 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)
    33 :   0.749394 :      + 3690 L4Cerenkov::PostStepDoIt(G4Track const&, G4Step const&)








Attempt to get iteration started without alignment : using direct key geocache + primary persisting
----------------------------------------------------------------------------------------------------

::

    ckm--(){ ckm-cd ; ./go.sh ; } 

        ## 1st executable : setup + save + bi-simulate ( but not instrumented) 
        ## 
        ##     Cerenkov minimal sets up geometry+primaries 
        ##     then persists to key geocache+gdml+primaries 
        ##

    ckm-okg4()
    {
        OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save
    }

        ##  2nd executable : compute, fully instrumented gorilla  
        ##
        ##      OKG4Test picks up geocache+GDML+primaries from the key geocache 
        ##      and proceeds to bi-simulate in compute mode
        ##

    ckm-okg4-load()
    {
        OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --load --envkey --embedded
    }

        ##  2nd executable again : load+viz 
        ##  
        ##      OKG4Test (with no G4) just loads event + geocache for visualization 
        ##


Immediate Issues
-----------------

* Lots of bad flags : skipped some asserts in CRecorder/CPhoton to complete

  * FIXED :doc:`OKG4Test_CPhoton_badflag` was using default G4OpBoundaryProcess when need the custom one

* CPU side indexing aborted, fails to create the phosel + recsel for G4 

  * FIXED :doc:`OKG4Test_OpticksEvent_indexPhotonsCPU_assert` by direct recording whether dynamic 
    in OpticksEvent and using that to decide on resize via setNumPhotons

* no hits from G4 : SD-LV ASSOCIATION DIDNT SURVIVE THE CACHE/GDML ?? 

  * :doc:`OKG4Test_no_G4_hits` actually now have in G4, but not reflected into OpticksEvent yet 

* OK hits lost, following the lv name changes 

  * fixed by consistent use of GDML pointer suffixed names  
  * :doc:`OKG4Test_no_OK_hits_again` FIXED at ckm (precache, directly converted) level at least 


* listed under torch, need a new "primaries" source code ? 

* FIXED longstanding issue of mixed timestamp event dirs when changes make eg the hits buffer go away,
  when now getting zero hits : cause is that NPY has problems with saving empties, so it just skips 



::

    epsilon:torch blyth$ np.py 
    /private/tmp/blyth/opticks/evt/g4live/torch
           ./Opticks.npy : (33, 1, 4) 

         ./-1/report.txt : 31 
           ./-1/idom.npy : (1, 1, 4) 
           ./-1/fdom.npy : (3, 1, 4) 
             ./-1/gs.npy : (5, 6, 4) 
             ./-1/no.npy : (152, 4, 4) 

             ./-1/rx.npy : (76, 10, 2, 4) 
             ./-1/ox.npy : (76, 4, 4) 
             ./-1/ph.npy : (76, 1, 2) 

    ./-1/20180819_214856/report.txt : 31 

          ./1/report.txt : 38 
            ./1/idom.npy : (1, 1, 4) 
            ./1/fdom.npy : (3, 1, 4) 
              ./1/gs.npy : (5, 6, 4) 
              ./1/no.npy : (152, 4, 4) 

              ./1/rx.npy : (76, 10, 2, 4) 
              ./1/ox.npy : (76, 4, 4) 
              ./1/ph.npy : (76, 1, 2) 

              ./1/ps.npy : (76, 1, 4) 
              ./1/rs.npy : (76, 10, 1, 4)     
              ./1/ht.npy : (9, 4, 4) 

    ./1/20180819_214856/report.txt : 38 
    epsilon:torch blyth$ 






New SourceCode for primaries needed ?::

    139 void CG4Ctx::initEvent(const OpticksEvent* evt)
    140 {
    141     _ok_event_init = true ;
    142     _photons_per_g4event = evt->getNumPhotonsPerG4Event() ;
    143     _steps_per_photon = evt->getMaxRec() ;
    144     _record_max = evt->getNumPhotons();   // from the genstep summation
    145     _bounce_max = evt->getBounceMax();
    146 
    147     const char* typ = evt->getTyp();
    148     _gen = OpticksFlags::SourceCode(typ);
    149     assert( _gen == TORCH || _gen == G4GUN  );
    150 
    151     LOG(info) << "CG4Ctx::initEvent"
    152               << " photons_per_g4event " << _photons_per_g4event
    153               << " steps_per_photon " << _steps_per_photon
    154               << " gen " << _gen
    155               ;
    156 }




Opticks to Opticks comparison between the two executables
------------------------------------------------------------

Issues: 

* different gensteps counts 


::

    epsilon:natural blyth$ abe-;abe-np
    A
    /private/tmp/blyth/opticks/evt/g4live/natural/1
            ./report.txt : 38 
                ./ps.npy :           (75, 1, 4) : bfeaba3dd698bbfedfebd6520c021a8e 
                ./ht.npy :            (9, 4, 4) : c5ffaeb58b17fd7d86091e1a37639cc9 
                ./rx.npy :       (75, 10, 2, 4) : 8948398d9830c5801e8ec3f80da0deed 
              ./fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 
                ./ox.npy :           (75, 4, 4) : ce7c87e0fcfd0109157bfd563b9e4290 
                ./gs.npy :            (4, 6, 4) : dafb2ce485c2005d9f361a2ccff44aa7 
                ./rs.npy :       (75, 10, 1, 4) : a4669c1e1366cb2c6b12c2491fecc796 
                ./ph.npy :           (75, 1, 2) : cf6319848e7321ffba5bcb6db1d25774 
              ./idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c 
    B
    /private/tmp/blyth/opticks/evt/g4live/torch/1
            ./report.txt : 38 
                ./ps.npy :           (54, 1, 4) : a9c747a22ccb0b456894977848eac259 
                ./ht.npy :            (9, 4, 4) : 7636b58de5a7438e0b0a4c4e81821714 
                ./rx.npy :       (54, 10, 2, 4) : 47cfde6ba92435586b98995911b23f21 
              ./fdom.npy :            (3, 1, 4) : f8c7c318e47b0ccb2c29567e87d95e67 
                ./ox.npy :           (54, 4, 4) : 992649d6cdb39ba01d21673ba614109d 
                ./no.npy :           (94, 4, 4) : d267dc315561cfd8b4e23885b994622a 
                ./gs.npy :            (3, 6, 4) : fd366013195541d193cbe96ecf0632f9 
                ./rs.npy :       (54, 10, 1, 4) : 06a500365854a8cefe9276e034a41300 
                ./ph.npy :           (54, 1, 2) : 3af8d0394239ecf934d753532f8d98a9 
              ./idom.npy :            (1, 1, 4) : a910ad1008e847548261491f9ca73f9c 




Although an OK-OK comparisison it is relying on genstep collection from G4, twice::


    epsilon:natural blyth$ abe-;abe-gs
    import numpy as np, commands

    apath = "/tmp/blyth/opticks/evt/g4live/natural/1/gs.npy"
    bpath = "/tmp/blyth/opticks/evt/g4live/torch/1/gs.npy"

    print " abe-xx- comparing gs.npy between two dirs " 

    print "  ", commands.getoutput("date")
    print "a ", commands.getoutput("ls -l %s" % apath)
    print "b ", commands.getoutput("ls -l %s" % bpath)

    a = np.load(apath)
    b = np.load(bpath)

    print "a %s " % repr(a.shape)
    print "b %s " % repr(b.shape)



    print "\n\na0/b0 : id/parentId/materialId/numPhotons \n " 
    a0 = "a[:,0].view(np.int32)"
    b0 = "b[:,0].view(np.int32)"

    print a0, "\n"
    print eval(a0), "\n"

    print b0, "\n"
    print eval(b0), "\n"


    print "\n\na1/b1 : start position and time x0xyz, t0 \n" 
    a1 = "a[:,1]"
    b1 = "b[:,1]"

    print a1, "\n"
    print eval(a1), "\n"

    print b1, "\n"
    print eval(b1), "\n"


    print "\n\na2/b2 : deltaPosition, stepLength \n" 
    a2 = "a[:,2]"
    b2 = "b[:,2]"

    print a2, "\n"
    print eval(a2), "\n"

    print b2, "\n"
    print eval(b2), "\n"


    args: /opt/local/bin/ipython -i /tmp/blyth/opticks/abe/abe-gs.py
     abe-xx- comparing gs.npy between two dirs 
       Wed Aug 22 16:51:53 CST 2018
    a  -rw-r--r--  1 blyth  wheel  464 Aug 22 15:08 /tmp/blyth/opticks/evt/g4live/natural/1/gs.npy
    b  -rw-r--r--  1 blyth  wheel  368 Aug 22 15:22 /tmp/blyth/opticks/evt/g4live/torch/1/gs.npy
    a (4, 6, 4) 
    b (3, 6, 4) 


    a0/b0 : id/parentId/materialId/numPhotons 
     
    a[:,0].view(np.int32) 

    [[  1   1   7  10]
     [  1   1   7  57]
     [  1   1   7   7]
     [  1 101  11   1]] 

    b[:,0].view(np.int32) 

    [[ 1  1  7 10]
     [ 1  1  7 34]
     [ 1  1  7 10]] 



    a1/b1 : start position and time x0xyz, t0 

    a[:,1] 

    [[ 0.      0.      0.      0.    ]
     [ 0.3439 -0.0689 -0.0139  0.0012]
     [ 1.6663 -0.4712 -0.2608  0.0064]
     [73.0526 65.0379 94.895   0.4662]] 

    b[:,1] 

    [[ 0.      0.      0.      0.    ]
     [ 0.3439 -0.0689 -0.0139  0.0012]
     [ 1.6658 -0.4711 -0.2607  0.0064]] 



    a2/b2 : deltaPosition, stepLength 

    a[:,2] 

    [[ 0.3439 -0.0689 -0.0139  0.3579]
     [ 1.3224 -0.4023 -0.247   1.8766]
     [-0.254  -0.4309 -0.3052  0.7534]
     [ 0.1046  0.0216  0.0442  0.1275]] 

    b[:,2] 

    [[ 0.3439 -0.0689 -0.0139  0.3578]
     [ 1.3219 -0.4022 -0.2469  1.8757]
     [ 0.1766  0.4707  0.2677  0.7315]] 


    In [1]: 








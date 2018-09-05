OKX4Test_j1808_x4gen_csg_solids_survey
==============================================

* for context :doc:`OKX4Test_j1808`


Setup for survey of solids
--------------------------------------------------------------------------

OKX4Test : boot with GDML, direct convert to GGeo, persist to geocache with codegen 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    opticksdata- ; OKX4Test --gdmlpath $(opticksdata-j) --g4codegen


copy the key "spec" into OPTICKS_KEY envvar, add to .bash_profile/.bashrc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice the key "spec" output and copy it into OPTICKS_KEY envvar::

    2018-08-30 21:58:03.003 INFO  [4283586] [BOpticksResource::setupViaKey@392] BOpticksKey
                        spec  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
                     exename  : OKX4Test
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : 15cf540d9c315b7f5d0adc7c3907b922
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

Export that envvar when reusing this geometry. NB Opticks executables 
are currently only sensitive to OPTICKS_KEY envvar when the --envkey option is used.::

    export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922 

    epsilon:opticks blyth$ geocache-info

      OPTICKS_KEY     :  OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.15cf540d9c315b7f5d0adc7c3907b922
      geocache-keydir : /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1

    epsilon:opticks blyth$ geocache-kcd
    epsilon:1 blyth$ pwd
    /usr/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/15cf540d9c315b7f5d0adc7c3907b922/1


generate CMakeLists.txt and scripts to build the generated code into executables for every solid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    x4gen--


x4gen- shakedown
-------------------

::

    [blyth@localhost ~]$ x4gen-deep
     so:016 lv:016 rmx:11 bmx:02 soName: sFasteners0x4c01080                  ## looks too good to be true : must be loosing geometry ?? YES see Issue B
    ----------------------------- height greater than 7 skipped in kernel -------
     so:018 lv:018 rmx:04 bmx:04 soName: PMT_20inch_inner1_solid0x4cb3610
     so:019 lv:019 rmx:04 bmx:04 soName: PMT_20inch_inner2_solid0x4cb3870
     so:020 lv:020 rmx:03 bmx:03 soName: PMT_20inch_body_solid0x4c90e50
     so:021 lv:021 rmx:03 bmx:03 soName: PMT_20inch_pmt_solid0x4c81b40




g4codegen mini-review
-----------------------

* :doc:`g4codegen_review`


survey the 40 solids of j1808
-------------------------------------------------------------

::

    [blyth@localhost 1]$ cat solids.txt 
    written by GGeoGLTF::solidRecTable 
    num_solid 40
     so:000 lv:000 rmx:00 bmx:00 soName: Upper_LS_tube0x5b2e9f0
     so:001 lv:001 rmx:01 bmx:01 soName: Upper_Steel_tube0x5b2eb10
     so:002 lv:002 rmx:01 bmx:01 soName: Upper_Tyvek_tube0x5b2ec30
     so:003 lv:003 rmx:00 bmx:00 soName: Upper_Chimney0x5b2e8e0
     so:004 lv:004 rmx:00 bmx:00 soName: sBar0x5b34ab0
     so:005 lv:005 rmx:00 bmx:00 soName: sBar0x5b34920
     so:006 lv:006 rmx:00 bmx:00 soName: sModuleTape0x5b34790
     so:007 lv:007 rmx:00 bmx:00 soName: sModule0x5b34600
     so:008 lv:008 rmx:00 bmx:00 soName: sPlane0x5b34470
     so:009 lv:009 rmx:00 bmx:00 soName: sWall0x5b342e0
     so:010 lv:010 rmx:01 bmx:01 soName: sAirTT0x5b34000

           LV=10 x4gen-csg  
           
           looks like a box subtract a cylinder 
           with serious case of coincidence subtraction speckles

           issue A : probably easy fix, just grow the subtracted cylinder in correct direction 

           YEP : FIXED WITH A DELTA 

     so:011 lv:011 rmx:00 bmx:00 soName: sExpHall0x4bcd390
     so:012 lv:012 rmx:00 bmx:00 soName: sTopRock0x4bccfc0
     so:013 lv:013 rmx:01 bmx:01 soName: sTarget0x4bd4340
     so:014 lv:014 rmx:01 bmx:01 soName: sAcrylic0x4bd3cd0
     so:015 lv:015 rmx:01 bmx:01 soName: sStrut0x4bd4b80

     so:016 lv:016 rmx:11 bmx:02 soName: sFasteners0x4c01080

           LV=16 x4gen-csg  

            bizarre shape, raytrace get:
               evaluative_csg : perfect tree height 11 exceeds current limit 
            .. not using the balanced or balancing failed ?
            had to force quit the raytrace

            issue B : investigate balancing for this tree

            FIXED : BY GENERALIZING THE TREE BALANCING 


     so:017 lv:017 rmx:02 bmx:02 soName: sMask0x4ca38d0
           LV=17 x4gen-csg  observatory dome shape, polygonization failed, raytrace looks OK 


     so:018 lv:018 rmx:04 bmx:04 soName: PMT_20inch_inner1_solid0x4cb3610

           LV=18 x4gen-csg  

           wow : profligate use of a depth 4 tree (31 nodes)
           when a single node would do: ellipsoid with z range

           issue C : profligate CSG chop : fix is easy, just need to convince people to use sane CSG  


     so:019 lv:019 rmx:04 bmx:04 soName: PMT_20inch_inner2_solid0x4cb3870

           LV=19 x4gen-csg  

           speckle neck 
           also : profligate use of CSG intersection to chop the cathode off 

           issue C : profligate CSG chop : fix is easy, just need to convince people to use sane CSG  
           issue D : speckle neck : fix is easy, just need to convince people to use hyperboloid neck            



     so:020 lv:020 rmx:03 bmx:03 soName: PMT_20inch_body_solid0x4c90e50

           LV=20 x4gen-csg  

           speckle neck from the torus subtraction, 
           but this time the speckle disappears when closeup 
           and from some angles 

           issue D : speckle neck : fix is easy, just need to convince people to use hyperboloid neck            


     so:021 lv:021 rmx:03 bmx:03 soName: PMT_20inch_pmt_solid0x4c81b40

           LV=21 x4gen-csg  

           ditto : speckle neck 

           issue D : speckle neck : fix is easy, just need to convince people to use hyperboloid neck            

     so:022 lv:022 rmx:00 bmx:00 soName: sMask_virtual0x4c36e10
     so:023 lv:023 rmx:00 bmx:00 soName: PMT_3inch_inner1_solid_ell_helper0x510ae30
     so:024 lv:024 rmx:00 bmx:00 soName: PMT_3inch_inner2_solid_ell_helper0x510af10
     so:025 lv:025 rmx:00 bmx:00 soName: PMT_3inch_body_solid_ell_ell_helper0x510ada0
     so:026 lv:026 rmx:00 bmx:00 soName: PMT_3inch_cntr_solid0x510afa0
     so:027 lv:027 rmx:01 bmx:01 soName: PMT_3inch_pmt_solid0x510aae0
     so:028 lv:028 rmx:01 bmx:01 soName: sChimneyAcrylic0x5b310c0
     so:029 lv:029 rmx:00 bmx:00 soName: sChimneyLS0x5b312e0
     so:030 lv:030 rmx:01 bmx:01 soName: sChimneySteel0x5b314f0
     so:031 lv:031 rmx:00 bmx:00 soName: sWaterTube0x5b30eb0
     so:032 lv:032 rmx:00 bmx:00 soName: svacSurftube0x5b3bf50
     so:033 lv:033 rmx:00 bmx:00 soName: sSurftube0x5b3ab80
     so:034 lv:034 rmx:01 bmx:01 soName: sInnerWater0x4bd3660
     so:035 lv:035 rmx:01 bmx:01 soName: sReflectorInCD0x4bd3040
     so:036 lv:036 rmx:00 bmx:00 soName: sOuterWaterPool0x4bd2960
     so:037 lv:037 rmx:00 bmx:00 soName: sPoolLining0x4bd1eb0
     so:038 lv:038 rmx:00 bmx:00 soName: sBottomRock0x4bcd770
     so:039 lv:039 rmx:00 bmx:00 soName: sWorld0x4bc2350




issue A : sAirTT0x5b34000 : FIXED FOR ME 
--------------------------------------------

::

    LV=10 x4gen-csg  

::

     27 G4VSolid* make_solid()
     28 {
     29     G4VSolid* b = new G4Box("BoxsAirTT0x5b33e60", 24000, 24000, 2500) ; // 1

     //   these are half-lengths 

     30     G4VSolid* d = new G4Tubs("Cylinder0x5b33ef0", 0, 500, 2000, 0, 6.28319) ; // 1

     //   the 2000 is z-half-length 

     31  
     32     G4ThreeVector A(0.000000,0.000000,-500.000000);
     33     G4VSolid* a = new G4SubtractionSolid("sAirTT0x5b34000", b, d, NULL, A) ; // 0
     34     return a ;
     35 } 

     //  z:  2500 - 500 = 2000 voila z-coincidence 


delta-ize g4codegen/tests/x010.cc::

     26 // start portion generated by nnode::to_g4code 
     27 G4VSolid* make_solid()
     28 {
     29     
     30     double delta = 1. ; 
     31     
     32     G4VSolid* b = new G4Box("BoxsAirTT0x5b33e60", 24000.000000, 24000.000000, 2500.000000) ; // 1
     33     G4VSolid* d = new G4Tubs("Cylinder0x5b33ef0", 0.000000, 500.000000, delta+2000.000000, 0.000000, CLHEP::twopi) ; // 1
     34     
     35     G4ThreeVector A(0.000000,0.000000,-(delta+500.000000));
     36     G4VSolid* a = new G4SubtractionSolid("sAirTT0x5b34000", b, d, NULL, A) ; // 0
     37     return a ;
     38 } 

rebuild and viz, SPECKLE AVOIDED::

    x4gen-go
    LV=10 x4gen-csg  






issue B : sFasteners0x4c01080 : deep tree : FIXED BY GENERALIZING TREE BALANCING 
-----------------------------------------------------------------------------------

* :doc:`OKX4Test_sFasteners_generalize_tree_balancing`


::

     so:016 lv:016 rmx:11 bmx:02 soName: sFasteners0x4c01080

           LV=16 x4gen-csg  

            bizarre shape, raytrace get:
               evaluative_csg : perfect tree height 11 exceeds current limit 
            .. not using the balanced or balancing failed ?
            had to force quit the raytrace

            issue B : investigate balancing for this tree



Extract from the generated code::

    epsilon:1 blyth$ cat g4codegen/tests/x016.cc

    ...

    // start portion generated by nnode::to_g4code 
    G4VSolid* make_solid()
    { 
        G4VSolid* k = new G4Tubs("solidFasteners_down0x4bff9b0", 80.000000, 150.000000, 5.000000, 0.000000, CLHEP::twopi) ; // 10
        G4VSolid* m = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 10
        
        G4ThreeVector A(0.000000,125.000000,-70.000000);
        G4VSolid* j = new G4UnionSolid("solid_FastenersUnion0x4bffbf0", k, m, NULL, A) ; // 9
        G4VSolid* o = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 9
        
        G4ThreeVector B(88.388348,88.388348,-70.000000);
        G4VSolid* i = new G4UnionSolid("solid_FastenersUnion0x4bffdd0", j, o, NULL, B) ; // 8
        G4VSolid* q = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 8
        
        G4ThreeVector C(125.000000,0.000000,-70.000000);
        G4VSolid* h = new G4UnionSolid("solid_FastenersUnion0x4c00030", i, q, NULL, C) ; // 7
        G4VSolid* s = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 7
        
        G4ThreeVector D(88.388348,-88.388348,-70.000000);
        G4VSolid* g = new G4UnionSolid("solid_FastenersUnion0x4c00290", h, s, NULL, D) ; // 6
        G4VSolid* u = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 6
        
        G4ThreeVector E(0.000000,-125.000000,-70.000000);
        G4VSolid* f = new G4UnionSolid("solid_FastenersUnion0x4c004f0", g, u, NULL, E) ; // 5
        G4VSolid* w = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 5
        
        G4ThreeVector F(-88.388348,-88.388348,-70.000000);
        G4VSolid* e = new G4UnionSolid("solid_FastenersUnion0x4c00750", f, w, NULL, F) ; // 4
        G4VSolid* y = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 4
        
        G4ThreeVector G(-125.000000,-0.000000,-70.000000);
        G4VSolid* d = new G4UnionSolid("solid_FastenersUnion0x4c009b0", e, y, NULL, G) ; // 3
        G4VSolid* a1 = new G4Tubs("solidFasteners_Bolts0x4bffad0", 0.000000, 10.000000, 70.000000, 0.000000, CLHEP::twopi) ; // 3
        
        G4ThreeVector H(-88.388348,88.388348,-70.000000);
        G4VSolid* c = new G4UnionSolid("solid_FastenersUnion0x4c00c10", d, a1, NULL, H) ; // 2
        G4VSolid* c1 = new G4Tubs("solidFasteners_up0x4c01b50", 0.000000, 150.000000, 10.000000, 0.000000, CLHEP::twopi) ; // 2
        
        G4ThreeVector I(0.000000,0.000000,-140.000000);
        G4VSolid* b = new G4UnionSolid("solidFasteners20x4c00e30", c, c1, NULL, I) ; // 1
        G4VSolid* e1 = new G4Tubs("solidFasteners_up10x4bff890", 41.000000, 50.000000, 25.000000, 0.000000, CLHEP::twopi) ; // 1
        
        G4ThreeVector J(0.000000,0.000000,-165.000000);
        G4VSolid* a = new G4UnionSolid("sFasteners0x4c01080", b, e1, NULL, J) ; // 0
        return a ; 
    } 
    // end portion generated by nnode::to_g4code 






Originally 8 bolts and 2 plates and one rim?, one plate and the rim? has non-zero rmin, 
so: 8 + 1 + 2 + 2 = 13 


::

    2018-08-30 23:27:54.425 INFO  [4332762] [X4CSG::init@113] NTreeAnalyse height 11 count 25
                                                                                          un            

                                                                                  un              di    

                                                                          un          cy      cy      cy

                                                                  un          cy                        

                                                          un          cy                                

                                                  un          cy                                        

                                          un          cy                                                

                                  un          cy                                                        

                          un          cy                                                                

                  un          cy                                                                        

          di          cy                                                                                

      cy      cy                                           





issue C : profligate CSG chop : fix is easy, just need to convince people to use sane CSG  
---------------------------------------------------------------------------------------------



issue D : speckle neck : fix is easy, just need to convince people to use hyperboloid neck            
---------------------------------------------------------------------------------------------

TODO: rework tboolean-12-- using G4Hype in x4gen g4codegen/tests






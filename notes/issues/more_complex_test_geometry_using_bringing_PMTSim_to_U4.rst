more_complex_test_geometry_using_bringing_PMTSim_to_U4
=========================================================

* from :doc:`higher_stats_U4RecorderTest_cxs_rainbow_random_aligned_comparison`

* HMM: PMTSim has previously been used for purely geometry intersect tests in x4, gc
* NOW need to use it to create Rock/Water/PMT test setups to shoot input photons at random aligned manner
* HMM: ths means bringing PMTSim to u4/U4VolumeMaker 

* TODO: compare PMTSim PMT and Mask Managers with the lastest ones 


DONE : gx level translation and CX running of U4VolumeMaker::PV GEOM hama_body_log
-------------------------------------------------------------------------------------

* :doc:`gxs_high_level_translation_of_U4VolumeMaker_Geant4_geometry`


With the water rock border surface get simpler history as no BT into Rock and then AB 
--------------------------------------------------------------------------------------

* TODO : increase beam radius from only 49mm radius 
* DONE : resize geometry from halfsize 5000 to 1000 mm to suit input photon starting point and PMT dimensions

::

    epsilon:tests blyth$ ./U4RecorderTest.sh ana
    ...

    In [1]: cuss(t.seq[:,0])
    Out[1]: 
    CUSS([['w0', '          TO BT BT BT BT BT SA', '       147639501', '            7820'],
          ['w1', '          TO BT BT BT BT BT AB', '        80530637', '            1046'],
          ['w2', '                TO BT BR BT SA', '          576461', '             324'],
          ['w3', '    TO BT BT BT BR BT BT BT SA', '     37795646669', '             288'],
          ['w4', '                         TO AB', '              77', '             211'],
          ['w5', '       TO BT BT BT BT BT SC SA', '      2261568717', '              87'],
          ['w6', '                TO BT BR BT AB', '          314317', '              34'],
          ['w7', '                      TO BT AB', '            1229', '              31'],
          ['w8', '    TO BT BT BT BR BT BT BT AB', '     20615777485', '              31'],
          ['w9', '             TO BT BT BT BT AB', '         5033165', '              28'],
          ['w10', ' TO BT BT BT BT BR BT BT BT BT', '    879608253645', '              20'],
          ['w11', '                      TO BR SA', '            2237', '              18'],
          ['w12', '                      TO SC SA', '            2157', '              16'],
          ['w13', ' TO BT BT BT BR BT BR BT BT BT', '    879592459469', '              12'],
          ['w14', '       TO BT BT BT BT BT SC AB', '      1187826893', '               7'],
          ['w15', ' TO BT BT BT BR BT BT BT SC SA', '    578961525965', '               3'],
          ['w16', '          TO SC BT BT BT BT AB', '        80530541', '               2'],
          ['w17', '    TO BT BT BT BT BT SC SC SA', '     36084436173', '               2'],
          ['w18', '                      TO SC AB', '            1133', '               2'],
          ['w19', '    TO BT BR BR BT BT BT BT SA', '     37795707853', '               2'],
          ['w20', '                      TO BR AB', '            1213', '               2'],
          ['w21', '             TO BT BR BT SC SA', '         8833997', '               2'],




DONE : hama_body_log showing lots of BULK_ABSORB ? Did I omit the simplifying SA absorber surface ? YES
--------------------------------------------------------------------------------------------------------

::


    In [3]: t.base
    Out[3]: '/tmp/blyth/opticks/U4RecorderTest/ShimG4OpAbsorption_FLOAT_ShimG4OpRayleigh_FLOAT/hama_body_log'

    In [2]: cuss(t.seq[:,0])
    Out[2]: 
    CUSS([['w0', '       TO BT BT BT BT BT BT AB', '      1288490189', '            7645'],
          ['w1', '          TO BT BT BT BT BT AB', '        80530637', '            1046'],
          ['w2', '             TO BT BR BT BT AB', '         5032909', '             316'],
          ['w3', ' TO BT BT BT BR BT BT BT BT AB', '    329853422797', '             279'],
          ['w4', '                         TO AB', '              77', '             211'],
          ['w5', ' TO BT BT BT BT BT BR BT BT BT', '    879592525005', '              66'],
          ['w6', '    TO BT BT BT BT BT SC BT AB', '     20515179725', '              65'],
          ['w7', '    TO BT BT BT BT BT BR BT AB', '     20599065805', '              53'],
          ['w8', '                TO BT BR BT AB', '          314317', '              34'],
          ['w9', '                      TO BT AB', '            1229', '              31'],
          ['w10', '    TO BT BT BT BR BT BT BT AB', '     20615777485', '              31'],
          ['w11', '             TO BT BT BT BT AB', '         5033165', '              28'],
          ['w12', ' TO BT BT BT BT BT BR BT BR BT', '    875297557709', '              27'],
          ['w13', '       TO BT BT BT BT BT BR AB', '      1271712973', '              25'],
          ['w14', ' TO BT BT BT BT BR BT BT BT BT', '    879608253645', '              20'],
          ['w15', '                   TO BR BT AB', '           19645', '              18'],
          ['w16', '                   TO SC BT AB', '           19565', '              13'],
          ['w17', ' TO BT BT BT BR BT BR BT BT BT', '    879592459469', '              12'],

::

    In [6]: cseqhis_("TO BT BT BT BT BT BT AB")
    Out[6]: 1288490189

    In [8]: w = np.where(t.seq[:,0] == cseqhis_("TO BT BT BT BT BT BT AB") )[0] ; w
    Out[8]: array([   0,    2,    3,    4,    6, ..., 9993, 9994, 9995, 9996, 9997])

    In [11]: t.photon[w,0,2]      ## YEP: getting BULK_ABSORB at various small depths into the Rock 
    Out[11]: array([5000.002, 5000.001, 5000.001, 5000.001, 5000.   , ..., 5000.   , 5000.   , 5000.   , 5000.002, 5000.001], dtype=float32)

    In [12]: t.photon[w,0,2].min()
    Out[12]: 5000.0

    In [13]: t.photon[w,0,2].max()
    Out[13]: 5000.008

    In [14]: t.photon[w,0,2].shape
    Out[14]: (7645,)




DONE : use PMTSim geometry within u4/tests/U4VolumeMaker.sh and U4RecorderTest.sh 
------------------------------------------------------------------------------------

U4RecorderTest.sh::

     85 source ./IDPath_override.sh   
     86 # IDPath_override.sh : non-standard IDPath to allow U4Material::LoadOri to find material properties 
     87 # HMM probably doing nothing now that are using U4Material::LoadBnd ?
     88 
     89 #geom=BoxOfScintillator
     90 geom=RaindropRockAirWater
     91 export GEOM=${GEOM:-$geom}

     97 G4VPhysicalVolume* U4RecorderTest::Construct(){ return U4VolumeMaker::Make(); } // sensitive to GEOM envvar 


* DONE : pull out bits of RaindropRockAirWater geometry setup and incorporate into a generalized U4VolumeMaker::Wrap
  to allow putting anything inside a RockWater test box

  * rationalized U4VolumeMaker  


Prior usage of PMTSim
-----------------------

::

    epsilon:opticks blyth$ find . -name CMakeLists.txt -exec grep -l -H PMTSim {} \;
    ./extg4/CMakeLists.txt
    ./extg4/tests/CMakeLists.txt
    ./GeoChain/CMakeLists.txt
    ./g4ok/tests/CMakeLists.txt   ## PMTSim from g4ok just an idea, so far unimplemented

x4 : used PMTSim as source of geometry for G4 intersect tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

x4/CMakeLists.txt::

    222 if(PMTSim_standalone_FOUND)
    223     target_link_libraries( ${name} PUBLIC Opticks::PMTSim )
    224     target_compile_definitions( ${name} PUBLIC WITH_PMTSIM )
    225 endif()

x4/tests/CMakeLists.txt::

     08 set(PMTSIM_TEST_SOURCES
      9     X4IntersectSolidTest.cc
     10     X4IntersectVolumeTest.cc
     11     X4MeshTest.cc
     12     X4VolumeMakerTest.cc
     13 )

gc : used PMTSim as source of geometry for translation tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gc/tests/GeoChainVolumeTest.cc::

     22 int main(int argc, char** argv)
     23 {
     24     OPTICKS_LOG(argc, argv);
     25     const char* name_default = "hama_body_phys"  ;
     26     const char* name = SSys::getenvvar("GEOM", name_default );
     27 
     28     const G4VPhysicalVolume* pv = nullptr ;
     29 #ifdef WITH_PMTSIM
     30     pv = PMTSim::GetPV(name);
     31 #endif
     38     const char* argforced = "--allownokey --gparts_transform_offset" ;
     40     Opticks ok(argc, argv, argforced);
     41     ok.configure();
     42 
     43     GeoChain chain(&ok);
     44     chain.convertPV(pv);
     46     chain.save(name);
     48     return 0 ;
     49 }




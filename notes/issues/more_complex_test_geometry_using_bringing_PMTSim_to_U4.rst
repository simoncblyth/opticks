more_complex_test_geometry_using_bringing_PMTSim_to_U4
=========================================================

* from :doc:`higher_stats_U4RecorderTest_cxs_rainbow_random_aligned_comparison`

* HMM: PMTSim has previously been used for purely geometry intersect tests in x4, gc
* NOW need to use it to create Rock/Water/PMT test setups to shoot input photons at random aligned manner
* HMM: ths means bringing PMTSim to u4/U4VolumeMaker 

* TODO: compare PMTSim PMT and Mask Managers with the lastest ones 


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

    222 if(PMTSim_FOUND)
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




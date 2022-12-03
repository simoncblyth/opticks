test_build_without_juno_specifics_WITH_PMTFASTSIM_or_WITH_PMTSIM
===================================================================

Almost all Geant4 depending packages have optional dependency on JUNO specific PMTSim and PMTFastSim::

    epsilon:opticks blyth$ find . -name CMakeLists.txt -exec grep -H WITH_PMTFASTSIM {} \;
    ./extg4/CMakeLists.txt:      target_compile_definitions( ${name} PUBLIC WITH_PMTFASTSIM )
    ./GeoChain/CMakeLists.txt:   target_compile_definitions( ${name} PUBLIC WITH_PMTFASTSIM )
    ./u4/CMakeLists.txt:         target_compile_definitions( ${name} PUBLIC WITH_PMTFASTSIM PMTFASTSIM_STANDALONE )


    epsilon:opticks blyth$ find . -name CMakeLists.txt -exec grep -H WITH_PMTSIM {} \;
    ./extg4/CMakeLists.txt:             target_compile_definitions( ${name} PUBLIC WITH_PMTSIM )
    ./extg4/tests/CMakeLists.txt:       target_compile_definitions( ${TGT} PUBLIC WITH_PMTSIM )
    ./GeoChain/CMakeLists.txt:          target_compile_definitions( ${name} PUBLIC WITH_PMTSIM )

    ./g4ok/tests/CMakeLists.txt:        target_compile_definitions( ${TGT} PUBLIC WITH_PMTSIM )   ## OLD INACTIVE PKG 

    ./qudarap/tests/CMakeLists.txt:     target_compile_definitions( ${TGT} PRIVATE WITH_PMTSIM )
         ## HUH : WHY THIS DEPENDENCY ?
         ## its use of JPMT.h from QPMTTest.cc : so not Geant4 related

    ./u4/CMakeLists.txt:                target_compile_definitions( ${name} PUBLIC WITH_PMTSIM PMTSIM_STANDALONE )



om-clean does not uninstall, so manully delete the cmake folder to get WITH_PMTFASTSIM to not get set::


    jfs 
    om   ## to find the install directory 

    epsilon:PMTFastSim blyth$ rm -rf /usr/local/opticks/lib/cmake/pmtfastsim

    oo   # check full Opticks build



Do the same for WITH_PMTSIM::

    jps
    om

    epsilon:PMTSim blyth$ rm -rf /usr/local/opticks/lib/cmake/pmtsim
  







Custom4_must_stay_as_an_optional_external
===========================================


qudarap and u4 depend on Custom4::

    epsilon:opticks blyth$ opticks-fl Custom4
    ./qudarap/CMakeLists.txt
    ./qudarap/tests/QPMT_MockTest.sh
    ./qudarap/tests/QSim_MockTest.sh
    ./qudarap/tests/QPMT_Test.sh
    ./u4/CMakeLists.txt
    ./u4/tests/CMakeLists.txt
    ./u4/tests/U4Custom4Test.sh
    ./u4/tests/U4Custom4Test.cc
    ./examples/UseCustom4/go.sh
    ./examples/UseCustom4/CMakeLists.txt
    epsilon:opticks blyth$ 


::

    epsilon:qudarap blyth$ l ${OPTICKS_PREFIX}_externals/custom4/
    total 0
    0 drwxr-xr-x   4 blyth  staff  128 Jul  2 15:35 0.1.6
    0 drwxr-xr-x   5 blyth  staff  160 Jul  2 15:35 .
    0 drwxr-xr-x   4 blyth  staff  128 Jul  2 14:52 0.1.4
    0 drwxr-xr-x   4 blyth  staff  128 Jul  2 12:26 0.1.5
    0 drwxr-xr-x  27 blyth  staff  864 Jul  2 12:26 ..
    epsilon:qudarap blyth$ 




The Custom4 that is found depends on CMAKE_PREFIX_PATH::

    epsilon:opticks blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/custom4/0.1.6
    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
    /usr/local/opticks
    /usr/local/opticks/externals
    /usr/local/optix
    epsilon:opticks blyth$ 

To test building when Custom4 is not found change ~/.opticks_config commenting 
the opticks-prepend-prefix line::

     73 # standard 1042 
     74 opticks-prepend-prefix /usr/local/opticks_externals/clhep
     75 opticks-prepend-prefix /usr/local/opticks_externals/g4_1042
     76 
     77 #opticks-prepend-prefix /usr/local/opticks_externals/custom4/0.1.4   
     78 #opticks-prepend-prefix /usr/local/opticks_externals/custom4/0.1.5   ## last tag 
     79 opticks-prepend-prefix /usr/local/opticks_externals/custom4/0.1.6    ## next tag 
     80 

Then do a clean build::

    om-
    qu
    om-clean

HMM a complication is that PMTSim also depends on Custom4, so when 
removing Custom4 also need to remove PMTSim. 

And PMTSim is not so easy to remove because it gets installed 
into OPTICKS_PREFIX::

    epsilon:opticks blyth$ find . -name '*PMTSim*'
    ./geocache/G4OKPMTSimTest_nnvt_body_phys_g4live
    ./geocache/G4OKPMTSimTest_World_pv_g4live
    ./include/PMTSim
    ./include/PMTSim/PMTSim.hh
    ./include/PMTSim/PMTSimParamSvc
    ./include/PMTSim/PMTSimParamSvc/_PMTSimParamData.h
    ./include/PMTSim/PMTSimParamSvc/PMTSimParamData.h
    ./include/PMTFastSim/PMTSimParamSvc
    ./include/PMTFastSim/PMTSimParamSvc/_PMTSimParamData.h
    ./include/PMTFastSim/PMTSimParamSvc/PMTSimParamData.h
    ./lib/pkgconfig/PMTSim.pc
    ./lib/PMTSimTest
    ./lib/libPMTSim.dylib
    ./build/PMTSim
    ./build/PMTSim/libPMTSim.dylib
    ./build/PMTSim/CMakeFiles/PMTSim.dir
    ./build/PMTSim/CMakeFiles/PMTSim.dir/Users/blyth/junotop/junosw/Simulation/DetSimV2/PMTSim
    ./build/PMTSim/tests/PMTSimTest
    ./build/PMTSim/tests/CMakeFiles/PMTSimTest.dir
    ./build/PMTSim/tests/CMakeFiles/PMTSimTest.dir/PMTSimTest.cc.o
    ./build/PMTSim/PMTSim.pc
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ pwd
    /usr/local/opticks
    epsilon:opticks blyth$ 



Need to rearrange where PMTSim gets installed::

    epsilon:PMTSim blyth$ t om-cmake
    om-cmake () 
    { 
        local sdir=$1;
        local bdir=$PWD;
        [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000;
        local rc;
        cmake $sdir -G "$(om-cmake-generator)" -DCMAKE_BUILD_TYPE=$(opticks-buildtype) -DOPTICKS_PREFIX=$(om-prefix) -DCMAKE_INSTALL_PREFIX=$(om-prefix) -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules;
        rc=$?;
        return $rc
    }
    epsilon:PMTSim blyth$ 


Hmm that proves difficult, so for now kludge removal of PMTSim with::
  
    jps
    ./om_remove.sh 












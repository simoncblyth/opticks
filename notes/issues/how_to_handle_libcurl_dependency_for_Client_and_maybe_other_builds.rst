how_to_handle_libcurl_dependency_for_Client_and_maybe_other_builds
====================================================================

::

    (ok) A[blyth@localhost libcurl]$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    plog
    nljson


The above opticks managed externals are treated that way for convenience and
because they probably will not cause version conflicts, as they are not being 
commonly used by other packages. libcurl is widely used and available from the system 
in a version that is not new enough, so treating it as a managed external 
is not the best way.

Eventually would be best for the right version of libcurl to be accessed from::

   /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/

This suggests addition of OPTICKS_CURL_PREFIX so it can be found just 
like CUDA, OPTIX, GEANT4 are::

    (ok) A[blyth@localhost Modules]$ env | grep OPTICKS
    OPTICKS_CUDA_PREFIX=/usr/local/cuda-12.4
    OPTICKS_BUILDTYPE=Debug
    OPTICKS_ENVNOTE=/home/blyth/j/local.sh:lo
    OPTICKS_OPTIX_PREFIX=/cvmfs/opticks.ihep.ac.cn/external/OptiX_800
    OPTICKS_CONFIG=Debug
    OPTICKS_HOME=/home/blyth/opticks
    OPTICKS_COMPUTE_ARCHITECTURES=70,89
    OPTICKS_GEANT4_PREFIX=/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Geant4/10.04.p02.juno
    OPTICKS_PREFIX=/data1/blyth/local/opticks_Debug
    OPTICKS_COMPUTE_CAPABILITY=70
    OPTICKS_DOWNLOAD_CACHE=/cvmfs/opticks.ihep.ac.cn/opticks_download_cache
    OPTICKS_STTF_PATH=/data1/blyth/local/opticks_Debug/externals/imgui/imgui/extra_fonts/Cousine-Regular.ttf


And a temporary place to put it is::

    (ok) A[blyth@localhost Modules]$ l /cvmfs/opticks.ihep.ac.cn/external/
    total 10
    1 drwxrwxr-x. 6 cvmfs cvmfs  231 Jun  6  2025 .
    1 -rw-r--r--. 1 cvmfs cvmfs    0 Jun  6  2025 .cvmfscatalog
    1 drwxrwxr-x. 5 cvmfs cvmfs   84 Jun  6  2025 NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64
    1 drwxrwxr-x. 5 cvmfs cvmfs   84 Jun  6  2025 NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64
    1 drwxrwxr-x. 5 cvmfs cvmfs   84 Jun  6  2025 NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64
    1 drwxrwxr-x. 5 cvmfs cvmfs   84 Jun  6  2025 NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
    1 lrwxrwxrwx. 1 cvmfs cvmfs   37 Sep  3  2024 OptiX_800 -> NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64
    1 lrwxrwxrwx. 1 cvmfs cvmfs   37 Sep  3  2024 OptiX_770 -> NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64
    1 lrwxrwxrwx. 1 cvmfs cvmfs   37 Sep  3  2024 OptiX_760 -> NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64
    1 lrwxrwxrwx. 1 cvmfs cvmfs   37 Nov 12  2023 OptiX_750 -> NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64


OR can simply require addition to CMAKE_PREFIX_PATH::

     32 #extra=/home/blyth/miniconda3/envs/ok
     33 extra=/data1/blyth/local/opticks_Client/externals
     35 export CMAKE_PREFIX_PATH=red:green:blue:$extra


Actually the clean way to do that is to add a bashrc in the JUNOSW style that can be sourced::

     290 local_ok_externals()
     291 {
     292    : ~/j/local.sh
     293    : Xercesc + CLHEP + Geant4 + custom4 + Python + python-numpy versions matching JUNOSW
     294    : NB versions need to match those used by gitlab-ci build when using the gitlab cvmfs release
     295 
     296    local_unset
     297 
     298    local_c4_build  # setup c4 externals
     299    source $(local_c4_prefix)/bashrc
     300 
     301    source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/ExternalLibs/Python/3.11.10/bashrc
     302    source /cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J25.4.0/ExternalLibs/python-numpy/1.26.4/bashrc
     303 
     304    export VIP_MODE=ok_externals
     305    export VIP_DESC="Opticks JUNOSW externals in common manually setup from cvmfs : Xercesc + CLHEP + Geant4 + custom4 + Python + python-numpy"
     306 }


Essentially PATH CMAKE_PREFIX_PATH LD_LIBRARY_PATH::

     377 lob_client () 
     378 {
     379     lob  # Assuming this sets up your base Opticks env
     380 
     381     export OPTICKS_CONFIG=Client
     382 
     383     local extra=/data1/blyth/local/opticks_Client/externals
     384     export CMAKE_PREFIX_PATH=$extra:$CMAKE_PREFIX_PATH
     385     export LD_LIBRARY_PATH=$extra/lib:$LD_LIBRARY_PATH
     386     export PATH=$extra/bin:$PATH
     387 
     388     export OPTICKS_ENVNOTE=$BASH_SOURCE:$FUNCNAME
     389 }




Hmm libssl mixup causing many test fails::

    FAILS:  34  / 221   :  Mon Apr 27 17:27:28 2026  :  GEOM J26_1_1_opticks_Debug  
      1  /1   Test #1  : GDXMLTest.GDXMLTest                                     ***Failed                      0.01   
      1  /31  Test #1  : U4Test.Deprecated_U4PhotonInfoTest                      ***Failed                      0.01   
      2  /31  Test #2  : U4Test.U4TrackInfoTest                                  ***Failed                      0.01   
      3  /31  Test #3  : U4Test.U4TrackTest                                      ***Failed                      0.01   
      4  /31  Test #4  : U4Test.U4Custom4Test                                    ***Failed                      0.01   
      5  /31  Test #5  : U4Test.U4NistManagerTest                                ***Failed                      0.01   
      6  /31  Test #6  : U4Test.U4MaterialTest                                   ***Failed                      0.01   
      7  /31  Test #7  : U4Test.U4MaterialPropertyVectorTest                     ***Failed                      0.01   
      8  /31  Test #8  : U4Test.U4GDMLTest                                       ***Failed                      0.01   
      9  /31  Test #9  : U4Test.U4GDMLReadTest                                   ***Failed                      0.01   
      10 /31  Test #10 : U4Test.U4PhysicalConstantsTest                          ***Failed                      0.01   
      11 /31  Test #11 : U4Test.U4RandomTest                                     ***Failed                      0.01   
      12 /31  Test #12 : U4Test.U4UniformRandTest                                ***Failed                      0.01   
      13 /31  Test #13 : U4Test.U4EngineTest                                     ***Failed                      0.01   
      14 /31  Test #14 : U4Test.U4RandomMonitorTest                              ***Failed                      0.01   
      15 /31  Test #15 : U4Test.U4RandomArrayTest                                ***Failed                      0.01   
      16 /31  Test #16 : U4Test.U4VolumeMakerTest                                ***Failed                      0.01   
      17 /31  Test #17 : U4Test.U4LogTest                                        ***Failed                      0.01   
      18 /31  Test #18 : U4Test.U4RotationMatrixTest                             ***Failed                      0.01   
      19 /31  Test #19 : U4Test.U4TransformTest                                  ***Failed                      0.01   
      20 /31  Test #20 : U4Test.U4TraverseTest                                   ***Failed                      0.01   
      21 /31  Test #21 : U4Test.U4Material_MakePropertyFold_MakeTest             ***Failed                      0.01   
      22 /31  Test #22 : U4Test.U4Material_MakePropertyFold_LoadTest             ***Failed                      0.01   
      23 /31  Test #23 : U4Test.U4TouchableTest                                  ***Failed                      0.01   
      24 /31  Test #24 : U4Test.U4SurfaceTest                                    ***Failed                      0.01   
      25 /31  Test #25 : U4Test.U4SolidTest                                      ***Failed                      0.01   
      26 /31  Test #26 : U4Test.U4SolidMakerTest                                 ***Failed                      0.01   
      27 /31  Test #27 : U4Test.U4SensitiveDetectorTest                          ***Failed                      0.01   
      28 /31  Test #28 : U4Test.U4Debug_Test                                     ***Failed                      0.01   
      29 /31  Test #29 : U4Test.U4Hit_Debug_Test                                 ***Failed                      0.01   
      30 /31  Test #30 : U4Test.G4ThreeVectorTest                                ***Failed                      0.01   
      31 /31  Test #31 : U4Test.U4PhysicsTableTest                               ***Failed                      0.01   
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                                 ***Failed                      0.01   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test                   ***Failed                      0.01   



::

    U4SensitiveDetectorTest: /home/blyth/miniconda3/envs/ok/lib/libssl.so.3: version `OPENSSL_3.2.0' not found (required by /data1/blyth/local/opticks_Debug/lib/../externals/lib64/libcurl.so.4)
    /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh : FAIL from U4SensitiveDetectorTest

          Start 28: U4Test.U4Debug_Test
    28/31 Test #28: U4Test.U4Debug_Test ...........................***Failed    0.01 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/u4/tests
                    GEOM : J26_1_1_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh
              EXECUTABLE : U4Debug_Test
                    ARGS : 
    U4Debug_Test: /home/blyth/miniconda3/envs/ok/lib/libssl.so.3: version `OPENSSL_3.2.0' not found (required by /data1/blyth/local/opticks_Debug/lib/../externals/lib64/libcurl.so.4)
    /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh : FAIL from U4Debug_Test

          Start 29: U4Test.U4Hit_Debug_Test
    29/31 Test #29: U4Test.U4Hit_Debug_Test .......................***Failed    0.01 sec
                    HOME : /home/blyth
                     PWD : /data1/blyth/local/opticks_Debug/build/u4/tests
                    GEOM : J26_1_1_opticks_Debug
             BASH_SOURCE : /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh
              EXECUTABLE : U4Hit_Debug_Test
                    ARGS : 
    U4Hit_Debug_Test: /home/blyth/miniconda3/envs/ok/lib/libssl.so.3: version `OPENSSL_3.2.0' not found (required by /data1/blyth/local/opticks_Debug/lib/../externals/lib64/libcurl.so.4)
    /data1/blyth/local/opticks_Debug/bin/U4TestRunner.sh : FAIL from U4Hit_Debug_Test

          Start 30: U4Test.G4ThreeVectorTest





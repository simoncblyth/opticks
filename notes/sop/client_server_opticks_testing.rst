client_server_opticks_testing
================================


server
--------

CSGOptiX level server implemented using FastAPI python
together with nanobind to connect from python to C++ CSGOptiX instance.

::

    ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh



C++ client using libcurl : ~/np/tests/np_curl_test/np_curl_test.sh
---------------------------------------------------------------------

Client implemented using NP_CURL.h NP.hh and libcurl
This test client repeatedly uploads the same gensteps.


commandline client using curl : ~/np/tests/np_curl_test/np_curl_test.sh
-------------------------------------------------------------------------

::

    ~/np/tests/np_curl_test/np_curl_test.sh cli



BUILD_WITH_CUDA
----------------

::

    (ok) A[blyth@localhost opticks]$ find . -name CMakeLists.txt -exec grep -H BUILD_WITH_CUDA {} \;
    ./CSG/tests/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./CSG/tests/CMakeLists.txt:if(BUILD_WITH_CUDA) # TMP
    ./CSG/CMakeLists.txt:option(BUILD_WITH_CUDA "${name} Build with CUDA support" ON)  # default:ON/OFF
    ./CSG/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./CSG/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./CSG/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./CSG/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./CSG/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./g4cx/CMakeLists.txt:option(BUILD_WITH_CUDA "${name} Build with CUDA support" ON)  # default:ON/OFF
    ./g4cx/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./g4cx/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./g4cx/CMakeLists.txt:   message(STATUS "${CMAKE_CURRENT_LIST_FILE} : NOT-BUILD_WITH_CUDA so no CSGOptiX")
    ./g4cx/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./okconf/CMakeLists.txt:option(BUILD_WITH_CUDA "${name} Build with CUDA support" ON)  # default:ON/OFF
    ./okconf/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./okconf/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./qudarap/CMakeLists.txt:set(BUILD_WITH_CUDA ON)
    ./qudarap/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./qudarap/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/CMakeLists.txt:option(BUILD_WITH_CUDA "${name} Build with CUDA support" ON)  # default:ON/OFF
    ./sysrap/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/tests/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/tests/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./sysrap/tests/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./u4/tests/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./u4/CMakeLists.txt:option(BUILD_WITH_CUDA "${name} Build with CUDA support" ON)  # default:ON/OFF
    ./u4/CMakeLists.txt:if(BUILD_WITH_CUDA)
    ./u4/CMakeLists.txt:if(BUILD_WITH_CUDA)
    (ok) A[blyth@localhost opticks]$ 





NEXT : realistic client that collects gensteps from Geant4 and uses NP_CURL.h to upload them and download hits 
----------------------------------------------------------------------------------------------------------------

Difficulties:

* hit post-processing for localization (summary "muon" hits do not need this) 
* reduced dependency Opticks build : skipping CUDA, OptiX, CSGOptiX
* how to organize ? OJ interface can stay the same - just need some switch




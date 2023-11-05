okdist-tests-failing-for-lack-of-GEOM
=======================================


Issue
--------

Packaging up the tests (following okdist- workflow) and running them works::

    CTestTestfile.py $(opticks-bdir) --dest /tmp/tests   
    cd /tmp/tests
    ctest --output-on-error 
 
But running them lacks GEOM and other environment, 
so get lots of fails. 

Just the sysrap fails::

    epsilon:tests blyth$ pwd
    /usr/local/opticks/tests
    epsilon:tests blyth$ ctest --output-on-failure


    The following tests FAILED:
         95 - SysRapTest.SSimTest (INTERRUPT)
         96 - SysRapTest.SBndTest (INTERRUPT)
        101 - SysRapTest.SEnvTest_FAIL (Failed)
    Errors while running CTest
    epsilon:tests blyth$ 



HMM: opticks-t/om-test bash machinery does the env setup
How to do the equivalent withing ctest ? 


experiment with running ctests via a bash wrapper for env setup
-----------------------------------------------------------------

Using the below managed to get ctests to run via a bash script::

	sysrap/tests/STestRunner.sh

Testing with::

	sysrap/tests/SEnvTest_FAIL.cc
	sysrap/tests/SEnvTest_PASS.cc

Commands to test::

    st
    om-cd
    ctest -R SEnvTest_PASS --output-on-failure
    ctest -R SEnvTest_FAIL --output-on-failure


That also works from okdist-install-tests tree
-------------------------------------------------

::

    okdist-install-tests
    cd /usr/local/opticks/tests
    ctest -R SEnvTest_FAIL --output-on-failure  

BUT: notice lots of hardcoded absolute paths back to the src tree

/usr/local/opticks/tests/sysrap/tests/CTestTestfile.cmake::

    205 add_test(SysRapTest.STTFTest "STTFTest")
    206 set_tests_properties(SysRapTest.STTFTest PROPERTIES  _BACKTRACE_TRIPLES "/Users/blyth/opticks/sysrap/tests/CMakeLists.txt;16    8;add_test;/Users/blyth/opticks/sysrap/tests/CMakeLists.txt;0;")
    207 add_test(SysRapTest.SEnvTest_FAIL "/opt/local/bin/bash" "/Users/blyth/opticks/sysrap/tests/STestRunner.sh" "SEnvTest_FAIL")
    208 set_tests_properties(SysRapTest.SEnvTest_FAIL PROPERTIES  _BACKTRACE_TRIPLES "/Users/blyth/opticks/sysrap/tests/CMakeLists.t    xt;185;add_test;/Users/blyth/opticks/sysrap/tests/CMakeLists.txt;0;")
    209 add_test(SysRapTest.SEnvTest_PASS "/opt/local/bin/bash" "/Users/blyth/opticks/sysrap/tests/STestRunner.sh" "SEnvTest_PASS")
    210 set_tests_properties(SysRapTest.SEnvTest_PASS PROPERTIES  _BACKTRACE_TRIPLES "/Users/blyth/opticks/sysrap/tests/CMakeLists.t    xt;185;add_test;/Users/blyth/opticks/sysrap/tests/CMakeLists.txt;0;")

    

   


ctest environment setup for tests
------------------------------------

* https://www.scivision.dev/cmake-ctest-set-environment-variable/

::

   set_property(TEST bar PROPERTY ENVIRONMENT "FOO=1;BAZ=0")



* https://stackoverflow.com/questions/48954920/trouble-setting-environment-variables-for-ctest-tests

::

    ADD_TEST(NAME testPyMyproj
        COMMAND ${CMAKE_COMMAND} -E env
            LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
            ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_scripts/test_pyMyproj.py
    )


::

    find . -name CMakeLists.txt -exec grep -H add_test {} \;


    ./ok/tests/CMakeLists.txt:    add_test(${name}.${TGT} ${TGT})
    ./ok/tests/CMakeLists.txt:    #add_test(NAME ${name}.${TGT} COMMAND OpticksCTestRunner.sh --config $<CONFIGURATION> --exe $<TARGET_FILE:${TGT}> --remote-args --compute)
    ./integration/tests/CMakeLists.txt:    add_test(${name}.${TGT} ${TGT})
    ./integration/tests/CMakeLists.txt:add_test(${TEST} ${SCRIPT} --generateoverride 10000)
    ./extg4/tests/CMakeLists.txt:    add_test(${testname} ${TGT})
    ./extg4/tests/CMakeLists.txt:    add_test(${testname} ${TGT})


     18 foreach(SRC ${TESTS})
     19     get_filename_component(TGT ${SRC} NAME_WE)
     20     add_executable(${TGT} ${SRC})
     21 
     22     add_test(${name}.${TGT} ${TGT})
     23 
     24     #add_test(NAME ${name}.${TGT} COMMAND OpticksCTestRunner.sh --config $<CONFIGURATION> --exe $<TARGET_FILE:${TGT}> --rem    ote-args --compute)
     25     # cmake version 3.4.1 doesnt set the appropriate exe path but cmake version 3.5.2 does  
     26     # (avoid the complication by moving the smarts into the executable)
     27 
     28     #add_dependencies(check ${TGT})
     29 
     30     target_link_libraries(${TGT} OK)
     31     install(TARGETS ${TGT} DESTINATION lib)
     32 endforeach()


ok-undefined-reference-to-glfwSetJoystickCallback
====================================================

Trying to build with a new LOCAL_BASE on Gold, that fakes the lxslc/cluster paths::


     47 old_local_base(){
     48    export LOCAL_BASE=$HOME/local
     49 }   
     50 
     51 new_local_base(){
     52    export G=/hpcfs/juno/junogpu/blyth
     53    export LOCAL_BASE=$G/local
     54 }   
     55 
     56 #old_local_base
     57 new_local_base
     58 


Shows up this new issue which linking tests from ok proj::

    -- Build files have been written to: /hpcfs/juno/junogpu/blyth/local/opticks/build/ok
    === om-make-one : ok              /home/blyth/opticks/ok                                       /hpcfs/juno/junogpu/blyth/local/opticks/build/ok             
    Scanning dependencies of target OK
    [  7%] Building CXX object CMakeFiles/OK.dir/OK_LOG.cc.o
    [ 14%] Building CXX object CMakeFiles/OK.dir/OKMgr.cc.o
    [ 21%] Building CXX object CMakeFiles/OK.dir/OKPropagator.cc.o
    [ 28%] Linking CXX shared library libOK.so
    [ 28%] Built target OK
    Scanning dependencies of target flagsTest
    Scanning dependencies of target TrivialTest
    Scanning dependencies of target LogTest
    Scanning dependencies of target OKTest
    Scanning dependencies of target OTracerTest
    [ 35%] Building CXX object tests/CMakeFiles/flagsTest.dir/flagsTest.cc.o
    [ 42%] Building CXX object tests/CMakeFiles/LogTest.dir/LogTest.cc.o
    [ 50%] Building CXX object tests/CMakeFiles/OKTest.dir/OKTest.cc.o
    [ 57%] Building CXX object tests/CMakeFiles/OTracerTest.dir/OTracerTest.cc.o
    [ 64%] Building CXX object tests/CMakeFiles/TrivialTest.dir/TrivialTest.cc.o
    [ 71%] Linking CXX executable flagsTest
    /home/blyth/local/opticks/lib64/libOGLRap.so: undefined reference to `glfwSetJoystickCallback'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/flagsTest] Error 1
    make[1]: *** [tests/CMakeFiles/flagsTest.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....
    [ 78%] Linking CXX executable OTracerTest

::

    [blyth@localhost opticks]$ opticks-f glfwSetJoystickCallback
    ./oglrap/gleq.h:    glfwSetJoystickCallback(gleq_joystick_callback);



Try commenting oglrap/gleq.h::

    357 GLEQDEF void gleqInit(void)
    358 {
    359     glfwSetMonitorCallback(gleq_monitor_callback);
    360 #if GLFW_VERSION_MINOR >= 2
    361     //glfwSetJoystickCallback(gleq_joystick_callback);
    362 #endif
    363 }

But still the same::

    opticks-cd
    rm include/OGLRap/gleq.h
    om-cleaninstall oglrap:

Hmm gleq- is not being treated as a proper external as its just a single header
which is potentially causing issues of stomping my above comment out.

Nope that might cause other things, but probably just pilot error (session with stale env),
as doing it again and it works::

    [blyth@localhost opticks]$ grep Joystick include/OGLRap/gleq.h
        //glfwSetJoystickCallback(gleq_joystick_callback);

The cleaninstall runs until hitting a separate xercesc issue in x4::

    [ 97%] Linking CXX executable X4PhysicalVolume2Test
    /usr/bin/ld: warning: libicui18n.so.58, needed by /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so, not found (try using -rpath or -rpath-link)
    /usr/bin/ld: warning: libicuuc.so.58, needed by /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so, not found (try using -rpath or -rpath-link)
    /usr/bin/ld: warning: libicudata.so.58, needed by /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so, not found (try using -rpath or -rpath-link)
    /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so: undefined reference to `uset_getSerializedSet_58'
    /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so: undefined reference to `u_toupper_58'
    /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so: undefined reference to `UCNV_FROM_U_CALLBACK_SUBSTITUTE_58'
    /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so: undefined reference to `ucnv_setFromUCallBack_58'
    /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so: undefined reference to `ucnv_getMinCharSize_58'
    /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so: undefined reference to `ucnv_openU_58'


Going back to the old LOCAL_BASE and doing the below works::

    om-cleaninstall oglrap:

This older build is using system xercesc ?






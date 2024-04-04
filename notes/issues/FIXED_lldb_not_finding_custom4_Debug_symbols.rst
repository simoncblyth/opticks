FIXED_lldb_not_finding_custom4_Debug_symbols
==============================================

* FIX BY DELETING BUILD DIR ? GO FIGURE ? 


Attemping to check call sites, reveals using Release Custom4 on Darwin::

     BP=G4ParticleChange::ProposeVelocity ~/o/g4cx/tests/G4CXTest_raindrop_CPU.sh 

So rebuild custom4 in Debug mode into labelled dir::

    psilon:u4 blyth$ l /usr/local/opticks_externals/custom4_Debug/0.1.9/lib/libCustom4.dylib 
    536 -rwxr-xr-x  1 blyth  staff  271916 Apr  4 15:53 /usr/local/opticks_externals/custom4_Debug/0.1.9/lib/libCustom4.dylib

    epsilon:u4 blyth$ l /usr/local/opticks_externals/custom4_Release/0.1.9/lib/libCustom4.dylib 
    232 -rwxr-xr-x  1 blyth  staff  117884 Apr  4 15:53 /usr/local/opticks_externals/custom4_Release/0.1.9/lib/libCustom4.dylib

Change config, and clean build opticks against that::

    epsilon:u4 blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/custom4_Debug/0.1.9
    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
    /usr/local/opticks_Debug
    /usr/local/opticks_Debug/externals
    /usr/local/optix

    epsilon:u4 blyth$ echo $DYLD_LIBRARY_PATH | tr ":" "\n"
    /usr/local/opticks_externals/custom4_Debug/0.1.9/lib
    /usr/local/opticks_externals/g4_1042/lib
    /usr/local/opticks_externals/clhep/lib
    /usr/local/opticks_externals/xercesc/lib
    /usr/local/opticks_externals/boost/lib
    /usr/local/opticks_Debug/lib
    /usr/local/opticks_Debug/externals/lib
    /usr/local/cuda/lib
    epsilon:u4 blyth$ 
        


but not seeing the symbols, and notice gdb error::

    (lldb) b G4ParticleChange::ProposeVelocity
    error: libCustom4.dylib debug map object file '/private/tmp/blyth/customgeant4/build/CMakeFiles/Custom4.dir/C4OpBoundaryProcess.cc.o' has changed (actual time is 2024-04-04 15:53:37.000000000, debug map time is 2024-04-04 15:53:18.000000000) since this executable was linked, file will be ignored
    Breakpoint 1: 3 locations.
    (lldb) b


Bizarre. Why is lldb looking there ? 
Delete the build dir and it works as expected::

    epsilon:issues blyth$ rm -rf /private/tmp/blyth/customgeant4/build

Also change to not using same dir for Release and Debug builds. 



client_flexibility_test_fails
==============================



::

    FAILS:  5   / 220   :  Fri Apr 24 18:14:50 2026  :  GEOM J26_1_1_opticks_Debug  
      7  /22  Test #7  : QUDARapTest.QEvt_Lifecycle_Test                         ***Failed                      0.33   
      19 /22  Test #19 : QUDARapTest.QSimWithEventTest                           ***Failed                      3.39   
      21 /22  Test #21 : QUDARapTest.QEvtTest                                    ***Failed                      0.33   
      3  /4   Test #3  : CSGOptiXTest.CSGOptiXRenderTest                         ***Failed                      3.05   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test                   ***Failed                      3.19   


::

     eg 12274(31497) with(without) server running
     opticks-vram-free(){  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits ; }


After stop the server to free up VRAM::

    FAILS:  2   / 220   :  Fri Apr 24 18:35:45 2026  :  GEOM J26_1_1_opticks_Debug  
      3  /4   Test #3  : CSGOptiXTest.CSGOptiXRenderTest                         ***Failed                      3.04   
      2  /2   Test #2  : G4CXTest.G4CXOpticks_setGeometry_Test                   ***Failed                      3.18   



SEventConfig::HasDevice NO causing cx null
---------------------------------------------


::

    Reading symbols from CSGOptiXRenderTest...
    Starting program: /data1/blyth/local/opticks_Debug/lib/CSGOptiXRenderTest 
    [Thread debugging using libthread_db enabled]
    Using host libthread_db library "/lib64/libthread_db.so.1".
    [Detaching after vfork from child process 1295883]
    2026-04-24 18:40:35.619 FATAL [1295879] [CSGOptiX::Create@371] SEventConfig::HasDevice NO 
    SGLM::initView VIEW [-] load_interpolated_view NO  interpolated_view.brief -

    Program received signal SIGSEGV, Segmentation fault.
    0x000000000045e648 in std::vector<unsigned int, std::allocator<unsigned int> >::size (this=0x40) at /usr/include/c++/11/bits/stl_vector.h:919
    919	      { return size_type(this->_M_impl._M_finish - this->_M_impl._M_start); }
    Missing separate debuginfos, use: dnf debuginfo-install glibc-2.34-168.el9_6.23.x86_64
    (gdb) bt
    #0  0x000000000045e648 in std::vector<unsigned int, std::allocator<unsigned int> >::size (this=0x40) at /usr/include/c++/11/bits/stl_vector.h:919
    #1  0x0000000000417ad2 in CSGOptiXRenderTest::initArgs (this=0x7fffffffaf00) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:138
    #2  0x000000000041784a in CSGOptiXRenderTest::init (this=0x7fffffffaf00) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:100
    #3  0x00000000004177f4 in CSGOptiXRenderTest::CSGOptiXRenderTest (this=0x7fffffffaf00) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:92
    #4  0x00000000004180d4 in main (argc=1, argv=0x7fffffffb4c8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:183
    (gdb) f 4
    #4  0x00000000004180d4 in main (argc=1, argv=0x7fffffffb4c8) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:183
    183	    CSGOptiXRenderTest t;
    (gdb) f 3
    #3  0x00000000004177f4 in CSGOptiXRenderTest::CSGOptiXRenderTest (this=0x7fffffffaf00) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:92
    92	    init();
    (gdb) f 2
    #2  0x000000000041784a in CSGOptiXRenderTest::init (this=0x7fffffffaf00) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:100
    100	    initArgs();
    (gdb) f 1
    #1  0x0000000000417ad2 in CSGOptiXRenderTest::initArgs (this=0x7fffffffaf00) at /home/blyth/opticks/CSGOptiX/tests/CSGOptiXRenderTest.cc:138
    138	    unsigned num_select = cx->solid_selection.size();
    (gdb) p cx
    $1 = (CSGOptiX *) 0x0
    (gdb) 

    quit
    A debugging session is active.

        Inferior 1 [process 1295879] will be killed.

    Quit anyway? (y or n) y

    Fri Apr 24 06:41:26 PM CST 2026
    (ok) A[blyth@localhost tests]$ 






    ok) A[blyth@localhost tests]$ ./G4CXOpticks_setGeometry_Test.sh
    ./G4CXOpticks_setGeometry_Test.sh : GEOM J26_1_1_opticks_Debug : no geomscript
                       BASH_SOURCE : ./G4CXOpticks_setGeometry_Test.sh 
                               arg : info_run_ana 
                              SDIR :  
                              GEOM : J26_1_1_opticks_Debug 
                           savedir : /home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug 
                              FOLD : /home/blyth/.opticks/GEOM/J26_1_1_opticks_Debug 
                               bin : G4CXOpticks_setGeometry_Test 
                        geomscript :  
                            script : G4CXOpticks_setGeometry_Test.py 
                            origin :  
    ./GXTestRunner.sh - use externaly set GEOM CFBaseFromGEOM
                    HOME : /home/blyth
                     PWD : /home/blyth/opticks/g4cx/tests
                    GEOM : J26_1_1_opticks_Debug
    J26_1_1_opticks_Debug_GDMLPathFromGEOM : 
             BASH_SOURCE : ./GXTestRunner.sh
              EXECUTABLE : G4CXOpticks_setGeometry_Test
                    ARGS : 
    2026-04-24 18:42:45.445 INFO  [1296033] [main@16] [SetGeometry
    2026-04-24 18:42:46.421 FATAL [1296033] [CSGOptiX::Create@371] SEventConfig::HasDevice NO 
    G4CXOpticks_setGeometry_Test: /home/blyth/opticks/g4cx/G4CXOpticks.cc:411: static SSimulator* G4CXOpticks::CreateSimulator(CSGFoundry*): Assertion `cx' failed.
    ./GXTestRunner.sh: line 51: 1296033 Aborted                 (core dumped) $EXECUTABLE $@
    ./GXTestRunner.sh : FAIL from G4CXOpticks_setGeometry_Test
    ./G4CXOpticks_setGeometry_Test.sh : run error
    (ok) A[blyth@localhost tests]$ 
    (ok) A[blyth@localhost tests]$ 





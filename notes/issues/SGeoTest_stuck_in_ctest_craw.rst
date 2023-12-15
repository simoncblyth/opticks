SGeoTest_stuck_in_ctest_craw
===============================

ctest trying to run a removed test::

    epsilon:issues blyth$ opticks-f SGeoTest 
    ./sysrap/SGeo.hh:    ./sysrap/tests/SGeoTest.cc:#include "SGeo.hh"
    epsilon:opticks blyth$ 


Thats because are running the installed tests not the from the build dir tests.

    /home/blyth/junotop/ExternalLibs/opticks/head/tests/sysrap/tests/CTestTestfile.cmake::


::

    104 okjob-ctest()
    105 {
    106    cd $OPTICKS_PREFIX/tests
    107    pwd
    108    echo === $BASH_SOURCE $FUNCNAME $PWD
    109    which ctest
    110    ctest -N
    111    ctest --output-on-failure
    112 }



DONE : what is doing the test install ? : opticks-full/opticks-install-extras/opticks-install-tests
---------------------------------------------------------------------------------------------------

* opticks-full does that but thats not run often 
* as normally update build with oo : the tests installation is missed


::

    2005 opticks-full()
    2006 {
    2007     local msg="=== $FUNCNAME :"
    2008     local rc 
    2009 
    2010     opticks-info
    2011     [ $? -ne 0 ] && echo $msg ERR from opticks-info && return 1
    2012 
    2013     opticks-full-externals
    2014     [ $? -ne 0 ] && echo $msg ERR from opticks-full-externals && return 2
    2015 
    2016     opticks-full-make
    2017     [ $? -ne 0 ] && echo $msg ERR from opticks-full-make && return 3
    2018 
    2019     opticks-install-extras
    2020     [ $? -ne 0 ] && echo $msg ERR from opticks-install-extras && return 4
    2021 
    2022     opticks-cuda-capable
    2023     rc=$?
    2024     if [ $rc -eq 0 ]; then
    2025         echo $msg detected GPU proceed with opticks-full-prepare
    2026         opticks-full-prepare
    2027         rc=$?
    2028         [ $rc -ne 0 ] && echo $msg ERR from opticks-full-prepare && return 5
    2029     else
    2030         echo $msg detected no CUDA cabable GPU - skipping opticks-full-prepare
    2031         rc=0
    2032     fi
    2033     return 0
    2034 }




::

    epsilon:issues blyth$ opticks-f opticks-install-extras
    ./opticks.bash:opticks-install-extras
    ./opticks.bash:    opticks-install-extras
    ./opticks.bash:    [ $? -ne 0 ] && echo $msg ERR from opticks-install-extras && return 4
    ./opticks.bash:opticks-install-extras()
    epsilon:opticks blyth$ 




okdist-- mentions non existing okdist-create-tests  

::

    205    okdist-install-tests 
    206         Creates $(opticks-dir)/tests populated with CTestTestfile.cmake files 

::

    epsilon:production blyth$ opticks-f CTestTestfile.cmake
    ./bin/CMakeLists.py:        print("# Outputs to stdout the form of a toplevel CTestTestfile.cmake ", file=fp)
    ./bin/CMakeLists.py:        print("#    opticks-deps --testfile 1> $(opticks-bdir)/CTestTestfile.cmake ", file=fp)
    ./bin/CMakeLists.py:    parser.add_argument(     "--testfile", action="store_true", help="Generate to stdout a CTestTestfile.cmake with all subdirs" ) 
    ./bin/CTestTestfile.py:CTestTestfile.cmake files which list unit tests are copied 
    ./bin/CTestTestfile.py:A top level CTestTestfile.cmake composed of top level subdirs is added, 
    ./bin/CTestTestfile.py:    Copying CTestTestfile.cmake files from buildtree /home/blyth/local/opticks/build into a new destination tree /tmp/tests 
    ./bin/CTestTestfile.py:    write testfile to /tmp/tests/CTestTestfile.cmake 
    ./bin/CTestTestfile.py:    NAME = "CTestTestfile.cmake"
    ./bin/CTestTestfile.py:    parser.add_argument(     "root",  nargs=1, help="Base directory in which to look for CTestTestfile.cmake " )
    ./bin/okdist.bash:        Creates $(opticks-dir)/tests populated with CTestTestfile.cmake files 
    epsilon:opticks blyth$ 


::

    epsilon:opticks blyth$ git l  bin/CTestTestfile.py
    commit 01602625ab238fc42757485b4bc7cdd9626a5465
    Author: Simon C Blyth <simoncblyth@gmail.com>
    Date:   Sun Nov 5 20:06:34 2023 +0800

        rearrange ctests to run via bash scripts such as STestRunner.sh as that plays better with  okdist-install-tests installed ctests by avoiding the need for the om-testenv-push om-testenv-pop

    M       bin/CTestTestfile.py

    commit c052ffe7a46e1da18db0684ec0f93fc29f16ce73
    Author: Simon Blyth <simoncblyth@gmail.com>
    Date:   Thu Oct 10 22:24:55 2019 +0800

        opticks-site rearrange to allow overriding the opticks-site-release to for example a new candidate or opticks-dir for a source release

    M       bin/CTestTestfile.py

    commit 824441cf06648aadfbfe3b9207b3106e536c19b7
    Author: Simon Blyth <simoncblyth@gmail.com>
    Date:   Mon Sep 16 23:29:01 2019 +0800

        find way to install the CTest tests with bin/CTestTestfile.py, running them as user simon from the exploded binary tarball with ctest now stands at 21/411 fails, see notes/issues/shakedown-running-from-binary-dist.rst

    A       bin/CTestTestfile.py
    epsilon:opticks blyth$ 




Issue 
------

Not fixed by clean build::

    .       Start  26: SysRapTest.SDigestTest
     26/202 Test  #26: SysRapTest.SDigestTest ...................................   Passed    0.02 sec
            Start  27: SysRapTest.SDigestNPTest
     27/202 Test  #27: SysRapTest.SDigestNPTest .................................   Passed    0.02 sec
            Start  28: SysRapTest.SCFTest
     28/202 Test  #28: SysRapTest.SCFTest .......................................   Passed    0.02 sec
            Start  29: SysRapTest.SGeoTest
    Could not find executable SGeoTest
    Looked in the following places:
    SGeoTest
    SGeoTest
    Release/SGeoTest
    Release/SGeoTest
    Debug/SGeoTest
    Debug/SGeoTest
    MinSizeRel/SGeoTest
    MinSizeRel/SGeoTest
    RelWithDebInfo/SGeoTest
    RelWithDebInfo/SGeoTest
    Deployment/SGeoTest
    Deployment/SGeoTest
    Development/SGeoTest
    Development/SGeoTest
    Unable to find executable: SGeoTest
     29/202 Test  #29: SysRapTest.SGeoTest ......................................***Not Run   0.00 sec
            Start  30: SysRapTest.SCurandStateTest
     30/202 Test  #30: SysRapTest.SCurandStateTest ..............................   Passed    0.02 sec
            Start  31: SysRapTest.PLogTest
     31/202 Test  #31: SysRapTest.PLogTest ......................................   Passed    0.02 sec
            Start  32: SysRapTest.SLOG_Test


::

     34    SPairVecTest.cc
     35    SDigestTest.cc
     36    SDigestNPTest.cc
     37 
     38    SCFTest.cc
     39    SCurandStateTest.cc
     40 
     41   
     42    PLogTest.cc
     43    SLOG_Test.cc


Thats because are running the installed tests not from the build dir. 

/home/blyth/junotop/ExternalLibs/opticks/head/tests/sysrap/tests/CTestTestfile.cmake::

    add_test(SysRapTest.SDigestNPTest "SDigestNPTest")
    set_tests_properties(SysRapTest.SDigestNPTest PROPERTIES  _BACKTRACE_TRIPLES "/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;186;add_test;/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;0;")
    add_test(SysRapTest.SCFTest "SCFTest")
    set_tests_properties(SysRapTest.SCFTest PROPERTIES  _BACKTRACE_TRIPLES "/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;186;add_test;/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;0;")
    add_test(SysRapTest.SGeoTest "SGeoTest")
    set_tests_properties(SysRapTest.SGeoTest PROPERTIES  _BACKTRACE_TRIPLES "/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;186;add_test;/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;0;")
    add_test(SysRapTest.SCurandStateTest "SCurandStateTest")
    set_tests_properties(SysRapTest.SCurandStateTest PROPERTIES  _BACKTRACE_TRIPLES "/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;186;add_test;/data/blyth/junotop/opticks/sysrap/tests/CMakeLists.txt;0;")
    add_test(SysRapTest.PLogTest "PLogTest")




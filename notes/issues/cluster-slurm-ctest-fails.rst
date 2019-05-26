cluster-slurm-ctest-fails FIXED
=================================

Issue resolved : pilot error 
------------------------------

Looked like some kinda problem with slurm running of ctest executables. 

* all executables using OptiX failed when run via ctest
* but without ctest they worked
 
Misleading because ctest runs executables from the build tree but
only installed executables are in PATH.

Note that even though there are no available GPUs on build machine,
this kind of issue also manifested there, and is more easily diagnosed
as dont need to go via batch system.
This is because lib loading happens prior to failing for lack of GPU.

Turned out to be much more prosaic, simply am opticks installation with incompletely 
applied change to the OptiX library location, which was fixed by a clean install.   

* :doc:`lxslc-build-for-cluster-stuck-on-old-optix-install-dir`

In the process moved from mixed /afs and /hpcfs to pure /hpcfs for all Opticks paths 
as it avoids an om-cd fail and builds faster.


Installed tests worked without problem
---------------------------------------

::

  oxrap-
  oxrap-tests
  oxrap-tests-run


job script extract
---------------------

::

    #! /bin/bash -l

    #SBATCH --partition=gpu
    #SBATCH --qos=debug
    #SBATCH --account=junogpu
    #SBATCH --job-name=scb
    #SBATCH --ntasks=1
    #SBATCH --output=/hpcfs/juno/junogpu/blyth/out/%j.out
    #SBATCH --mem-per-cpu=20480
    #SBATCH --gres=gpu:v100:8

    ...

    oxrap-
    oxrap-bcd
    cd tests
    which ctest
    ctest --output-on-failure


Hmm all failing to load optix libs::

    bench.py --name geocache-machinery
    Namespace(digest=None, exclude=None, include=None, metric='launchAVG', name='geocache-machinery', other='prelaunch000', resultsdir='$OPTICKS_RESULTS_PREFIX/results', since=None)
    ()
    bench.py --name geocache-machinery
    /afs/ihep.ac.cn/users/b/blyth/g/local/bin/ctest
    Test project /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests
          Start  1: OptiXRapTest.OContextCreateTest
     1/19 Test  #1: OptiXRapTest.OContextCreateTest ..............***Failed    0.01 sec 
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/OContextCreateTest: error while loading shared libraries: liboptix.so.6.0.0: cannot open shared object file: No such file or directory

          Start  2: OptiXRapTest.OScintillatorLibTest
     2/19 Test  #2: OptiXRapTest.OScintillatorLibTest ............***Failed    0.01 sec 
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/OScintillatorLibTest: error while loading shared libraries: liboptix.so.6.0.0: cannot open shared object file: No such file or directory

          Start  3: OptiXRapTest.LTOOContextUploadDownloadTest
     3/19 Test  #3: OptiXRapTest.LTOOContextUploadDownloadTest ...***Failed    0.01 sec 
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/LTOOContextUploadDownloadTest: error while loading shared libraries: liboptix.so.6.0.0: cannot open shared object file: No such file or directory




Workaround
-------------
    
::

    # Note that ctest running uses the test executables from the build tree
    # not the installed executables which find libs via their RPATH.
    # Normally ctest manages to arrange that build tree test executables find their libs,
    # but somehow this fails to work with slurm batch running.
    # The workaround is to temporarily set the library path to find optix libs. 

    LD_LIBRARY_PATH=$LOCAL_BASE/opticks/externals/OptiX/lib64 ctest --output-on-failure

    LD_LIBRARY_PATH=$LOCAL_BASE/opticks/externals/OptiX/lib64 opticks-t



Maybe can avoid this by getting the RPATH set for executables in the build tree

* https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

Its already there.



Because dont have ssh access to cluster nodes, need to use core files to see backtraces
------------------------------------------------------------------------------------------

1. navigate to tests directory of build tree with failing tests 

::

   oxrap-bcd
   cd tests

Buildtree executables, logs and cores are all there::

    [blyth@lxslc701 out]$ oxrap-bcd
    [blyth@lxslc701 optixrap]$ pwd
    /hpcfs/juno/junogpu/blyth/local/opticks/build/optixrap
    [blyth@lxslc701 optixrap]$ cd tests
    [blyth@lxslc701 tests]$ l
    total 233752
    -rw-r--r--  1 blyth dyw     11175 May 26 21:24 interpolationTest.log
    -rw-r--r--  1 blyth dyw     21261 May 26 21:24 eventTest.log
    -rw-r--r--  1 blyth dyw      3121 May 26 21:23 downloadTest.log
    ...
    -rw-r--r--  1 blyth dyw      3490 May 26 21:23 boundaryTest.log
    -rw-r--r--  1 blyth dyw     18886 May 26 21:23 textureTest.log
    -rw-r--r--  1 blyth dyw      4116 May 26 21:23 bufferTest.log
    -rw-------  1 blyth dyw 444395520 May 26 21:22 core.99578
    -rw-r--r--  1 blyth dyw       801 May 26 21:22 Roots3And4Test.log
    -rw-------  1 blyth dyw 133009408 May 26 21:22 core.99517
    -rw-r--r--  1 blyth dyw       709 May 26 21:22 intersectAnalyticTest.log
     ...
    -rwxr-xr-x  1 blyth dyw    455416 May 26 20:57 boundaryLookupTest
    -rwxr-xr-x  1 blyth dyw    455024 May 26 20:57 boundaryTest
    -rwxr-xr-x  1 blyth dyw    492664 May 26 20:57 bufferTest
    -rwxr-xr-x  1 blyth dyw    414672 May 26 20:57 downloadTest
    -rwxr-xr-x  1 blyth dyw    473136 May 26 20:57 eventTest
    -rwxr-xr-x  1 blyth dyw    484872 May 26 20:57 interpolationTest
    -rwxr-xr-x  1 blyth dyw    390040 May 26 20:57 intersectAnalyticTest
    -rwxr-xr-x  1 blyth dyw    430496 May 26 20:57 LTOOContextUploadDownloadTest


Thats a surprise the RPATH is set and includes /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX/lib64::

    [blyth@lxslc701 tests]$ chrpath Roots3And4Test
    Roots3And4Test: RPATH=/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap:/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX/lib64:/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64:/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/lib:/usr/local/cuda/lib64:


Perhaps this means the problem was due to the change in the OptiX lib, that was incompletely reflected 
in the install. And was fixed by the clean install ?

* :doc:`lxslc-build-for-cluster-stuck-on-old-optix-install-dir`

YES, confirmed this. Can now run tests without doing anything special, just the below in the job script::

   opticks-t 



*file* tells you which core goes with which executable::

    [blyth@lxslc701 tests]$ file core.*
    core.99517: ELF 64-bit LSB core file x86-64, version 1 (SYSV), SVR4-style, from '/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/intersectAna', real uid: 20836, effective uid: 20836, real gid: 208, effective gid: 208, execfn: '/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/intersectAnalyticTest', platform: 'x86_64'
    core.99578: ELF 64-bit LSB core file x86-64, version 1 (SYSV), SVR4-style, from '/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/Roots3And4Te', real uid: 20836, effective uid: 20836, real gid: 208, effective gid: 208, execfn: '/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/Roots3And4Test', platform: 'x86_64'
    [blyth@lxslc701 tests]$ 




using the core
-----------------

* inconsistency warnings are from moving from mixed /afs and /hpcfs addressing to pure /hpcfs 

::

    [blyth@lxslc701 tests]$ gdb intersectAnalyticTest core.99517

    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-110.el7
    Copyright (C) 2013 Free Software Foundation, Inc.
    License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
    This is free software: you are free to change and redistribute it.
    There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
    and "show warranty" for details.
    This GDB was configured as "x86_64-redhat-linux-gnu".
    For bug reporting instructions, please see:
    <http://www.gnu.org/software/gdb/bugs/>...
    Reading symbols from /hpcfs/juno/junogpu/blyth/local/opticks/build/optixrap/tests/intersectAnalyticTest...done.

    warning: core file may not match specified executable file.
    [New LWP 99517]

    Using host libthread_db library "/usr/lib64/libthread_db.so.1".
    Core was generated by `/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/intersectAna'.
    Program terminated with signal 6, Aborted.
    #0  0x00002baa709b4207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-34.el7.x86_64 libcom_err-1.42.9-12.el7_5.x86_64 libgcc-4.8.5-28.el7_5.1.x86_64 libicu-50.1.2-15.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-28.el7_5.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-17.el7.x86_64
    (gdb) bt
    #0  0x00002baa709b4207 in raise () from /lib64/libc.so.6
    #1  0x00002baa709b58f8 in abort () from /lib64/libc.so.6
    #2  0x00002baa701bf7d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00002baa701bd746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00002baa701bd773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00002baa701bd993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x0000000000407221 in optix::ContextObj::checkError (this=0x131df50, code=RT_ERROR_FILE_NOT_FOUND)
        at /hpcfs/juno/junogpu/blyth/local/opticks/externals/OptiX_600/include/optixu/optixpp_namespace.h:2178
    #7  0x00002baa65278554 in optix::ContextObj::createProgramFromPTXFile (this=0x131df50, 
        filename="/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/intersectAnalyticTest_generated_intersect_analytic_torus_test.cu.ptx", 
        program_name="intersect_analytic_torus_test") at /hpcfs/juno/junogpu/blyth/local/opticks/externals/OptiX_600/include/optixu/optixpp_namespace.h:2549
    #8  0x00002baa65277f03 in OptiXTest::init (this=0x1329070, context=...) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/optixrap/OptiXTest.cc:57
    #9  0x00002baa65277d31 in OptiXTest::OptiXTest (this=0x1329070, context=..., cu=0x40cd08 "intersect_analytic_torus_test.cu", raygen_name=0x1016dc0 "intersect_analytic_torus_test", 
        exception_name=0x40cd76 "exception", buildrel=0x40cd51 "optixrap/tests", cmake_target=0x40cd60 "intersectAnalyticTest")
        at /afs/ihep.ac.cn/users/b/blyth/g/opticks/optixrap/OptiXTest.cc:43
    #10 0x0000000000405eac in main (argc=1, argv=0x7ffc89f08208) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/optixrap/tests/intersectAnalyticTest.cc:43
    (gdb) 

    (gdb) f 7
    #7  0x00002baa65278554 in optix::ContextObj::createProgramFromPTXFile (this=0x131df50, 
        filename="/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/intersectAnalyticTest_generated_intersect_analytic_torus_test.cu.ptx", 
        program_name="intersect_analytic_torus_test") at /hpcfs/juno/junogpu/blyth/local/opticks/externals/OptiX_600/include/optixu/optixpp_namespace.h:2549
    2549        checkError( rtProgramCreateFromPTXFile( m_context, filename.c_str(), program_name.c_str(), &program ) );
    (gdb) l
    2544      }
    2545    
    2546      inline Program ContextObj::createProgramFromPTXFile( const std::string& filename, const std::string& program_name )
    2547      {
    2548        RTprogram program;
    2549        checkError( rtProgramCreateFromPTXFile( m_context, filename.c_str(), program_name.c_str(), &program ) );
    2550        return Program::take(program);
    2551      }
    2552    
    2553      inline Program ContextObj::createProgramFromPTXFiles( const std::vector<std::string>& filenames, const std::string& program_name )
    (gdb) 




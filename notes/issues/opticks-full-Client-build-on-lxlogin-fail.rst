opticks-full-Client-build-on-lxlogin-fail
===========================================

::

    ssh L
    lo_client       ## this fails due to lack of setup - as opticks-full not done for Client config
    opticks-full

    ## HMM thats messy - add opticks-full-client ? 



Issue : looks like opticks-install-tests is unaware of Client config skipping qudarap and CSGOptiX packages ?
--------------------------------------------------------------------------------------------------------------

::

    -- Installing: /hpcfs/juno/junogpu/blyth/local/opticks_Client/cmake/Modules/OpticksCXXFlags.cmake
    -- Installing: /hpcfs/juno/junogpu/blyth/local/opticks_Client/lib/OpticksCMakeModulesTest
    -- Set runtime path of "/hpcfs/juno/junogpu/blyth/local/opticks_Client/lib/OpticksCMakeModulesTest" to "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64"
    === opticks-full-make : DONE Thu May 7 11:33:22 AM CST 2026
    === opticks-install-extras : install cmake/Modules
                FUNCNAME : opticks-install-cmake-modules 
                    home : /hpcfs/juno/junogpu/blyth/opticks 
                    dest : /hpcfs/juno/junogpu/blyth/local/opticks_Client 
                  script : /hpcfs/juno/junogpu/blyth/local/opticks_Client/bin/CMakeModules.py 
    [2026-05-07 11:33:23,284] p2182449 {/hpcfs/juno/junogpu/blyth/local/opticks_Client/bin/CMakeModules.py:53} INFO - Copying from src /hpcfs/juno/junogpu/blyth/opticks/cmake/Modules to dst /hpcfs/juno/junogpu/blyth/local/opticks_Client/cmake/Modules 
    === opticks-install-extras : install ctest
                FUNCNAME : opticks-install-tests 
                    bdir : /hpcfs/juno/junogpu/blyth/local/opticks_Client/build 
                    dest : /hpcfs/juno/junogpu/blyth/local/opticks_Client/tests 
                  script : /hpcfs/juno/junogpu/blyth/local/opticks_Client/bin/CTestTestfile.py 
                    fold : /hpcfs/juno/junogpu/blyth 
    [2026-05-07 11:33:23,933] p2182707 {/hpcfs/juno/junogpu/blyth/opticks/bin/CMakeLists.py:198} INFO - home /hpcfs/juno/junogpu/blyth/opticks 
    [2026-05-07 11:33:24,869] p2182707 {/hpcfs/juno/junogpu/blyth/local/opticks_Client/bin/CTestTestfile.py:68} INFO - root /hpcfs/juno/junogpu/blyth/local/opticks_Client/build 
    [2026-05-07 11:33:24,869] p2182707 {/hpcfs/juno/junogpu/blyth/local/opticks_Client/bin/CTestTestfile.py:69} INFO - projs ['okconf', 'sysrap', 'CSG', 'qudarap', 'gdxml', 'u4', 'CSGOptiX', 'g4cx'] 
    Traceback (most recent call last):
      File "/hpcfs/juno/junogpu/blyth/local/opticks_Client/bin/CTestTestfile.py", line 137, in <module>
        top = bt.filtercopy( dest )
              ^^^^^^^^^^^^^^^^^^^^^
      File "/hpcfs/juno/junogpu/blyth/local/opticks_Client/bin/CTestTestfile.py", line 85, in filtercopy
        shutil.copytree( src, dst,  symlinks=False, ignore=self ) 
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/cvmfs/juno.ihep.ac.cn/el9_amd64_gcc11/Release/J24.2.0/ExternalLibs/Python/3.11.10/lib/python3.11/shutil.py", line 571, in copytree
        with os.scandir(src) as itr:
             ^^^^^^^^^^^^^^^
    FileNotFoundError: [Errno 2] No such file or directory: '/hpcfs/juno/junogpu/blyth/local/opticks_Client/build/qudarap'
    === opticks-full : detected no CUDA cabable GPU - skipping opticks-full-prepare


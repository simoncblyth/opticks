packaging-opticks-and-externals-for-use-on-gpu-cluster
========================================================

Strategy
----------

1. DONE : rearrange the position of libraries such as OptiX to make packaging simpler
2. DONE : develop python + bash scripts to make tarball, okdist-- 

NOTE for publishing to cvmfs see hcvmfs-vi
--------------------------------------------


whats left for release ?
-------------------------

* scan- is handy and fairly standalone, want to get it into the release

  * interate on that from "su - simon" account first 

* structure of includes ?

  * probably need to install flattened like /home/blyth/local/opticks/externals/include/Geant4/
  * not needed fpor CMake approach 
  * see examples/UseG4

* NO LONGER RELEVANT ? something to replace opticks-config

  * do this manually with bash functions inside opticks-release.bash ? opticks-release-config ?
  * OR : follow something like CMake pkgconfig : but for a multi-proj  
  * OR : python parse the CMake/BCM exported targets 
 
  * check it with simple non-CMake Makefile based building against the binary release using opticks-release-config
  * this is a stand in for CMT, as would rather not touch that 

* DONE : CMake based build against the release

  * what needs to be included in the distribution for this to work ? 
  * lib64/cmake has tree of .cmake with the exported targets
  * examples/Geant4/CerenkovMinimal builds from simon

* DONE : single top level setup bash function, bin/opticks-site.bash  

  * avoid cluster users duplicating the setup function, three source lines plus
  * easier management when need to change it for new release etc.. 
  * intended to be cluster local script  
  * once debugged, need a better path to put it in : /hpcfs/opticks/ ?

* NEXT : CerenkovMinimal still writing to some blackhole /tmp paths::

   opticks-site-demojob- > demojob.sh 
   sbatch demojob.sh
   so 
   grep /tmp nnnn.out  


* opticks-release-test from other users account, Yan, on GPU cluster

  * checking the single top level setup,  
  * permissions problems, output to appropriate places

* example: sources,scripts etc.. in the install ?  

  * THIS can be deferred, until CerenkovMinimal more operational 
  * installing source code feels plain wrong, because of no repository backing 
  * for now can just keep them in source tree and instruct users how to clone it if needed



DONE
-------

1. replace all use of CMAKE_INSTALL_PREFIX in the cmake/Modules/FindXX.cmake with OPTICKS_PREFIX
   and detect that automatically for FindXX.cmake modules that are picked up from install tree. 

   * this allows the cmake/Modules/FindXX.cmake to work from user projects, 
     where CMAKE_INSTALL_PREFIX is not OPTICKS_PREFIX 

   * this means cmake usage from source tree needs to specify -DOPTICKS_PREFIX=$(om-prefix) 
     as the auto determined one is opticks-home not prefix

2. cmake/Modules/OpticksBuildOptions.cmake : fixed runpath setup to use absolute paths when a foreign install is detected


CerenkovMinimal built against release, cluster batch run test
---------------------------------------------------------------

* fixed build issue of not finding OpticksXercesC
* runtime still writing into blackhole /tmp paths
* need to iterate on this

::

    sj # modify job
    sb # submit it 
    so # check output 

   



simon CerenkovMinimal test
---------------------------

* CMake based build completes from a release on fake /cvmfs but get runtime permission problem

::

    [simon@localhost ~]$ cp -r ~blyth/opticks/examples/Geant4/CerenkovMinimal .
    [simon@localhost CerenkovMinimal]$ ./go-release.sh 
    pfx /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg
    -- The C compiler identification is GNU 4.8.5
    -- The CXX compiler identification is GNU 4.8.5
    ...
    2019-09-29 16:51:24.815 INFO  [31256] [Opticks::loadOriginCacheMeta@1769]  gdmlpath 
    2019-09-29 16:51:24.816 INFO  [31256] [G4Opticks::translateGeometry@201] ) Opticks /opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1
    2019-09-29 16:51:24.816 INFO  [31256] [G4Opticks::translateGeometry@204] ( CGDML
    terminate called after throwing an instance of 'boost::filesystem::filesystem_error'
      what():  boost::filesystem::remove: Permission denied: "/opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml"
    ./go-release.sh: line 33: 31256 Aborted                 (core dumped) $exe
    [simon@localhost CerenkovMinimal]$ 

    [simon@localhost CerenkovMinimal]$ gdb /tmp/simon/opticks/examples/CerenkovMinimal/lib/CerenkovMinimal

    (gdb) bt
    #0  0x00007fffe3089207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe308a8f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe39987d5 in __gnu_cxx::__verbose_terminate_handler() () from /lib64/libstdc++.so.6
    #3  0x00007fffe3996746 in ?? () from /lib64/libstdc++.so.6
    #4  0x00007fffe3996773 in std::terminate() () from /lib64/libstdc++.so.6
    #5  0x00007fffe3996993 in __cxa_throw () from /lib64/libstdc++.so.6
    #6  0x00007fffeb62b309 in (anonymous namespace)::error(bool, boost::filesystem::path const&, boost::system::error_code*, std::string const&) () from /lib64/libboost_filesystem-mt.so.1.53.0
    #7  0x00007fffeb62b83f in (anonymous namespace)::remove_file_or_directory(boost::filesystem::path const&, boost::filesystem::file_type, boost::system::error_code*) () from /lib64/libboost_filesystem-mt.so.1.53.0
    #8  0x00007fffeb62c9a0 in boost::filesystem::detail::remove(boost::filesystem::path const&, boost::system::error_code*) () from /lib64/libboost_filesystem-mt.so.1.53.0
    #9  0x00007fffebb794a7 in boost::filesystem::remove (p=...) at /usr/include/boost/filesystem/operations.hpp:496
    #10 0x00007fffebb77df3 in BFile::RemoveFile (path=0xd70990 "/opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml", sub=0x0, name=0x0)
        at /home/blyth/opticks/boostrap/BFile.cc:653
    #11 0x00007ffff792127e in CGDML::Export (path=0xd70990 "/opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/27d088654714cda61096045ff5eacc02/1/g4ok.gdml", world=0x911ed0) at /home/blyth/opticks/cfg4/CGDML.cc:59
    #12 0x00007ffff7bd1466 in G4Opticks::translateGeometry (this=0x8b21f0, top=0x911ed0) at /home/blyth/opticks/g4ok/G4Opticks.cc:205
    #13 0x00007ffff7bd0819 in G4Opticks::setGeometry (this=0x8b21f0, world=0x911ed0, standardize_geant4_materials=true) at /home/blyth/opticks/g4ok/G4Opticks.cc:152
    #14 0x00000000004187d8 in RunAction::BeginOfRunAction (this=0x8dbeb0) at /home/simon/CerenkovMinimal/RunAction.cc:43
    #15 0x00007ffff41f42e5 in G4RunManager::RunInitialization (this=0x6f5b50) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:347
    #16 0x00007ffff41f3d0f in G4RunManager::BeamOn (this=0x6f5b50, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:272
    #17 0x0000000000419bc2 in G4::beamOn (this=0x7fffffffdaa0, nev=1) at /home/simon/CerenkovMinimal/G4.cc:81
    #18 0x0000000000419a7f in G4::G4 (this=0x7fffffffdaa0, nev=1) at /home/simon/CerenkovMinimal/G4.cc:69
    #19 0x0000000000409a40 in main (argc=1, argv=0x7fffffffdc18) at /home/simon/CerenkovMinimal/CerenkovMinimal.cc:26
    (gdb) 


Depends on bash enviromnent with::

    source /home/blyth/local/opticks/externals/opticks-envg4.bash
    source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/bin/opticks-release.bash
    source /opticks/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/opticks-sharedcache.bash

    # hmm not convenient to keep flipping this, how to detect when shared geocache is appropriate ?
    #unset OPTICKS_GEOCACHE_PREFIX

    export OPTICKS_DEFAULT_INTEROP_CVD=1   # GPU that is connected to the monitor for multi-gpu machines
    export PATH=/tmp/$USER/lib:$PATH


* had to split the shared cached envvar control into rngcache and geocache : 
  as normally the shared rngcache is appropriate but often (eg CerekovMinimal) 
  cannot use shared geocache : cause will try to write there 



ISSUES
----------

* currently install dir has no automatic cleaning, so for example
  old projects and headers languish there unless manually deleted before

* examples/Geant4/CerenkovMinimal/go.sh 

  needs access to OpticksBuildOptions.cmake and FindG4.cmake etc from  cmake/Modules

  * can i combine :  cmake/Modules with lib64/cmake ??  

    * decided against : simpler to keep generated and edited things separate


examples/Geant4/CerenkovMinimal/go.sh : CMake without source tree
---------------------------------------------------------------------

1. installed cmake/Modules to avoid use of opticks-home

2. FindGLM.cmake is using CMAKE_INSTALL_PREFIX : which doesnt 
   work when thats pointing elsewhere 



::

     29 go-cmake-0()
     30 {
     31    local sdir=$1
     32    cmake $sdir \
     33         -DCMAKE_BUILD_TYPE=Debug \
     34         -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
     35         -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     36         -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules
     37 }
     38 
     39 go-cmake-without-source-tree()
     40 {
     41    local sdir=$1
     42    cmake $sdir \
     43         -DCMAKE_BUILD_TYPE=Debug \
     44         -DCMAKE_PREFIX_PATH="$(opticks-prefix)/externals;$(opticks-prefix)" \
     45         -DCMAKE_INSTALL_PREFIX=/tmp/$FUNCNAME \
     46         -DCMAKE_MODULE_PATH=$(opticks-prefix)/cmake/Modules
     47 }
     48 




FIXED : RUNPATH ORIGIN setup not working : using absolute RUNPATH when user build detected
---------------------------------------------------------------------------------------------- 

* as executable not in expected place relative to libs 
* 

::

    -- Installing: /tmp/go-cmake-without-source-tree/lib/CerenkovMinimal
    -- Set runtime path of "/tmp/go-cmake-without-source-tree/lib/CerenkovMinimal" to "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64"
    [blyth@localhost CerenkovMinimal]$ 
    [blyth@localhost CerenkovMinimal]$ 
    [blyth@localhost CerenkovMinimal]$ /tmp/go-cmake-without-source-tree/lib/CerenkovMinimal
    /tmp/go-cmake-without-source-tree/lib/CerenkovMinimal: error while loading shared libraries: libG4OK.so: cannot open shared object file: No such file or directory


* if CMAKE_INSTALL_PREFIX does not match the determined or provided OPTICKS_PREFIX can change to absolute runtime path 




Tryinh to run from release missng BCM
---------------------------------------------

* fixed by installing : externals/share/bcm

::

    [blyth@localhost opticks]$ bcm-ls
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMConfig.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMDeploy.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMExport.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMFuture.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMIgnorePackage.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMInstallTargets.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMPkgConfig.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMProperties.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMSetupVersion.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMTest.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/BCMToSnakeCase.cmake
    /home/blyth/local/opticks/externals/share/bcm/cmake/version.hpp
    [blyth@localhost opticks]$ opticks-


Running from release fails to find G4
-----------------------------------------

* geant4 libs are excluded from the release, so 
  need to communicate the alt location to the build ? 




opticks-config
------------------

::

    [blyth@localhost bin]$ opticks-f opticks-config
    ./cmake/Modules/OpticksConfigureConfigScript.cmake:# - Script for configuring and installing opticks-config script
    ./cmake/Modules/OpticksConfigureConfigScript.cmake:# The opticks-config script provides an sh based interface to provide
    ./cmake/Modules/OpticksConfigureConfigScript.cmake:      ${CMAKE_SOURCE_DIR}/opticks-config.in
    ./cmake/Modules/OpticksConfigureConfigScript.cmake:      ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/opticks-config
    ./cmake/Modules/OpticksConfigureConfigScript.cmake:      ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/opticks-config
    ./cmake/Modules/OpticksConfigureConfigScript.cmake:  install(FILES ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/opticks-config
    ./cmake/Modules/inactive/FindOpticks.cmake:find_program(OPTICKS_CONFIG NAMES opticks-config
    ./cudarap/cudarap.bash:    opticks-configure
    ./okconf/CMakeLists.txt:# generate opticks-config sh script into lib dir
    ./oldopticks.bash:   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use opticks-configure to wipe build dir and re-configure && return  
    ./oldopticks.bash:opticks-configure()



release-test fail : tboolean- 
--------------------------------

::

    /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/bin/tboolean.sh: line 74: tboolean-: command not found


* try making tboolean.sh more standalone at accessing tboolean.bash 
* for release running do not want the full opticks- machinery  


release testing
-------------------

::

    user_setup()
    {
        export HOME=/hpcfs/juno/junogpu/$USER

        ## hmm this works avoiding afs permissions issues with original HOME
        ## but seems not a good idea as liable to confuse  
        ## TODO: switch all use of HOME to be sensitive to OPTICKS_USER_HOME with HOME as fallback default 
        ##      so can switch that 

        export TMP=$HOME/tmp
        ## /tmp is a black hole as not same filesystem on GPU cluster and gateway  


        source /hpcfs/juno/junogpu/blyth/local/opticks/externals/envg4.bash

        ##source /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/bin/release.bash  # real /cvmfs
        source /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/bin/release.bash

        source /hpcfs/juno/junogpu/blyth/opticks.ihep.ac.cn/sc/releases/OpticksSharedCache-0.0.0_alpha/bin/sharedcache.bash
    }






Naming the Opticks distribution
--------------------------------

* Name to include versions of gcc and Geant4.
* Not OptiX as will incorporate that in the dist, 
  so its covered by the Opticks version 
* optixrap for 6.5 and 7.0 need to be totally different




Excluding G4 from distro and getting it as a "foreign" external 
------------------------------------------------------------------------

* :doc:`glew-is-only-external-other-that-geant4-installing-into-lib64`
* Moved it from lib64 to lib, leaving only G4 


Excluding the G4 libs and data results means::

     55 tests failed out of 413

::

    [simon@localhost ~]$ G4OKTest
    G4OKTest: error while loading shared libraries: libG4Tree.so: cannot open shared object file: No such file or directory


* relocatable fix with g4-envg4


"simon" : Mockup environment of a foreign Geant4 install to check Opticks binary dist can work with that situation
------------------------------------------------------------------------------------------------------------------------------- 

* see scdist- 


Mockup usage with a foreign Geant4 install : ie one not installed as part of Opticks
----------------------------------------------------------------------------------------------

* see g4-envg4


CVMFS releases layout
--------------------------

Maybe like this::

    [blyth@lxslc701 releases]$ l /cvmfs/sft.cern.ch/lcg/releases/XercesC/3.1.3-b3bf1/x86_64-centos7-gcc9-opt/
    total 21
    drwxr-xr-x 3 cvmfs cvmfs 4096 Jul 13 00:05 lib
    -rw-r--r-- 1 cvmfs cvmfs    0 Jul 12 18:04 gen-post-install.log
    -rw-r--r-- 1 cvmfs cvmfs 1315 Jul 12 18:04 XercesC-env.sh
    drwxr-xr-x 2 cvmfs cvmfs 4096 Jul 12 18:04 logs
    -rw-r--r-- 1 cvmfs cvmfs   14 Jul 12 18:04 version.txt
    drwxr-xr-x 2 cvmfs cvmfs 4096 Jul 12 18:04 bin
    drwxr-xr-x 3 cvmfs cvmfs 4096 Jul 12 18:04 include
    [blyth@lxslc701 releases]$ 

::

    /cvmfs/opticks.ihep.ac.cn/ok/releases/Opticks/0.0.0-alpha/x86_64-centos7-gcc48-geant4_10_04_p02-dbg/


Issue : what to include in binary dist ?  
--------------------------------------------

* things needed to run opticks executables 

  * executables + libs + PTX + gl shaders : YES
  * installcache/PTX ? YES
  * installcache/RNG ? NO : DONE : relocated RNG to OPTICKS_SHARED_CACHE_PREFIX/rngcache/RNG
  * installcache/OKC ? NO : DONE : eliminated this using CMake custom command+target 
  * geocache ? NO : relocated to OPTICKS_SHARED_CACHE_PREFIX/geocache 
  * external libs 

    * libs assumed not to overlap with user (offline) : OptiX, yoctogl, ...   YES 
    * libs which offline depends on already (eg Geant4) : exclude them and bake versions into distro name 
    * what about boost libs ? try without : boost version into name ?
 
* directory tree of CTest files for unit testing of installed executables 

* bash and python scripts, to be collected into an installed "bin" dir 

  * things needed by scripts at runtime 
  * python "headers" .ini and .json in include   

* things needed to build against Opticks 

  * includes (all ? or a selection ? "public" headers )
  * opticks-config script 


Lots of the python assumes OPTICKS_HOME is available
-------------------------------------------------------

DONE : Eliminate installcache/OKC
-------------------------------------

The ini and json files in OKC are used from python, they are kinda the python equivalent
of includes.  They however cannot entirely be derived from includes.  

* it would be more convenient to derive these files during the build and install them 
  along with includes rather than current approach of requiring users to run an 
  executable at runtime

* DONE: added custom commands to optickscore/CMakeLists.txt to generate and install them 

::

    -- Installing: /home/blyth/local/opticks/include/OpticksCore/OpticksPhoton_Enum.ini
    -- Installing: /home/blyth/local/opticks/include/OpticksCore/OpticksFlags_Abbrev.json

::

    [blyth@localhost opticks]$ opticks-f OKC/
    ./ana/base.py:    def __init__(self, path="$OPTICKS_INSTALL_CACHE/OKC/GFlagIndexLocal.ini"):
    ./ana/base.py:        self.abbrev = Abbrev("$OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json")
    ./ana/base.py:        self.abbrev = Abbrev("$OPTICKS_INSTALL_CACHE/OKC/OpticksFlagsAbbrevMeta.json")
    [blyth@localhost opticks]$ 



Old way required users to run OpticksPrepareInstallCacheTest
-------------------------------------------------------------

Old way used Opticks::prepareInstallCache

   OpticksPrepareInstallCacheTest '$INSTALLCACHE_DIR/OKC'
   
::

    3203 void Opticks::prepareInstallCache(const char* dir)
    3204 {
    3205     if(dir == NULL) dir = m_resource->getOKCInstallCacheDir() ;
    3206     LOG(info) << ( dir ? dir : "NULL" )  ;
    3207     m_resource->saveFlags(dir);
    3208     m_resource->saveTypes(dir);
    3209 }

    1063 void OpticksResource::saveFlags(const char* dir)
    1064 {
    1065     OpticksFlags* flags = getFlags();
    1066     LOG(info) << " dir " << dir ;
    1067     flags->save(dir);
    1068 }

    439 void OpticksFlags::save(const char* installcachedir)
    440 {
    441     LOG(info) << installcachedir ;
    442     m_index->setExt(".ini");
    443     m_index->save(installcachedir);
    444     m_abbrev_meta->save( installcachedir, ABBREV_META_NAME );
    445 }


    1115 void OpticksResource::saveTypes(const char* dir)
    1116 {
    1117     LOG(info) << "OpticksResource::saveTypes " << dir ;
    1118 
    1119     Types* types = getTypes();
    1120     types->saveFlags(dir, ".ini");
    1121 }
    1122 


Arranged a CMake custom target/command to install to /home/blyth/local/opticks/include/OpticksCore/OpticksPhotonEnum.ini





Issue : setup for opticks executables to find libs (including externals)
-----------------------------------------------------------------------------

cmake/Modules/OpticksBuildOptions.cmake::

    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64")


Issue : setup for offline code to build and link against Opticks
---------------------------------------------------------------------

* offline still not using CMake, so need to revive the opticks-config script to serve up 
  locations of headers


Issue : how to test the setup : firstly without offline 
---------------------------------------------------------- 

* setup a non-CMake simple build that uses some Opticks libs to test
  getting the config from opticks-config

* create script to explode tarball and test with another user

* TODO: revive opticks-config for this


Issue : how to run unittests for checking the binary installation
------------------------------------------------------------------

* can ctest do this ?  Perhaps YES for sysrap anyhow.
* just need to propagate a tree of CTestTestfile.cmake
* suspect these can be hooked together (even across projects) with "subdirs" 

::

    [blyth@localhost tests]$ head -10 CTestTestfile.cmake
    # CMake generated Testfile for 
    # Source directory: /home/blyth/opticks/sysrap/tests
    # Build directory: /home/blyth/local/opticks/build/sysrap/tests
    # 
    # This file includes the relevant testing commands required for 
    # testing this directory and lists subdirectories to be tested as well.
    add_test(SysRapTest.SOKConfTest "SOKConfTest")
    add_test(SysRapTest.SArTest "SArTest")
    add_test(SysRapTest.SArgsTest "SArgsTest")
    add_test(SysRapTest.STimesTest "STimesTest")

    [blyth@localhost tests]$ tail -10 CTestTestfile.cmake
    add_test(SysRapTest.SSetTest "SSetTest")
    add_test(SysRapTest.STimeTest "STimeTest")
    add_test(SysRapTest.SASCIITest "SASCIITest")
    add_test(SysRapTest.SAbbrevTest "SAbbrevTest")
    add_test(SysRapTest.SEnvTest.red "SEnvTest" "SEnvTest_C" "--info")
    set_tests_properties(SysRapTest.SEnvTest.red PROPERTIES  ENVIRONMENT "SEnvTest_COLOR=red")
    add_test(SysRapTest.SEnvTest.green "SEnvTest" "SEnvTest_C" "--info")
    set_tests_properties(SysRapTest.SEnvTest.green PROPERTIES  ENVIRONMENT "SEnvTest_COLOR=green")
    add_test(SysRapTest.SEnvTest.blue "SEnvTest" "SEnvTest_C" "--info")
    set_tests_properties(SysRapTest.SEnvTest.blue PROPERTIES  ENVIRONMENT "SEnvTest_COLOR=blue")
    [blyth@localhost tests]$ 

::

    [blyth@localhost tests]$ cp CTestTestfile.cmake /tmp/ss/
    [blyth@localhost tests]$ pwd
    /home/blyth/local/opticks/build/sysrap/tests
       
    cd /tmp/ss ; ctest   ## worked

Ahha seems I did this before, but decided to stick with per-proj::

    opticks-deps --testfile 1> $(opticks-bdir)/CTestTestfile.cmake

::

    strace -o /tmp/strace.log -e open ctest 
    strace -f -o /tmp/strace.log -e open ctest    
    ## follow forks needed : some exe are listed by not all ?



opticksdata 
--------------

* aiming to eliminate this entirely, instead can move to admin users responsiblilty 
  to direct geocache creation to the GDML file 


OPTICKS_GEOCACHE_PREFIX : flexible way to direct Opticks executables to the base geocache directory 
------------------------------------------------------------------------------------------------------

* geocache is big and it changes on a different cycle to code, so must be separate from binary distro
* also want to be able to share the geocache between all users of the GPU cluster 
* envvar to point at the geocache base directory 

* hmm what about G4Opticks and flexibile running from live geometry 

  * compute digest to identify geometry and look for the geocache 
    relative to the base, the default with no envvar can be in users home



Running without geocache gives misleading error 
---------------------------------------------------------

* trys to fallback to loading from DAE, thats not what you want should instruct to run geocache-create with a gdml file as input 
  to create the geocahce  

::

    okdist-test

    2019-09-11 19:36:01.264 INFO  [417403] [Opticks::loadOriginCacheMeta@1688]  gdmlpath 
    2019-09-11 19:36:01.264 INFO  [417403] [OpticksHub::loadGeometry@521] [ /tmp/blyth/opticks/okdist-test/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
    2019-09-11 19:36:01.265 ERROR [417403] [GGeo::init@456]  idpath /tmp/blyth/opticks/okdist-test/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1 cache_exists 0 cache_requested 1 m_loaded_from_cache 0 m_live 0 will_load_libs 0
    2019-09-11 19:36:01.265 WARN  [417403] [OpticksColors::load@71] OpticksColors::load FAILED no file at  dir /tmp/blyth/opticks/okdist-test/opticksdata/resource/OpticksColors with name OpticksColors.json
    2019-09-11 19:36:01.266 ERROR [417403] [GGeo::loadFromG4DAE@624] GGeo::loadFromG4DAE START
    2019-09-11 19:36:01.266 INFO  [417403] [AssimpGGeo::load@162] AssimpGGeo::load  path NULL query all ctrl NULL importVerbosity 0 loaderVerbosity 0
    2019-09-11 19:36:01.266 FATAL [417403] [AssimpGGeo::load@174]  missing G4DAE path (null)
    2019-09-11 19:36:01.266 FATAL [417403] [GGeo::loadFromG4DAE@629] GGeo::loadFromG4DAE FAILED : probably you need to download opticksdata 
    OpSnapTest: /home/blyth/opticks/ggeo/GGeo.cc:633: void GGeo::loadFromG4DAE(): Assertion `rc == 0 && "G4DAE geometry file does not exist, try : opticksdata- ; opticksdata-- "' failed.
    Aborted (core dumped)
    -rw-rw-r--. 1 blyth blyth 11059217 Sep 11 11:32 /home/blyth/local/opticks/tmp/snap00000.ppm




Objective : test use of exploded binary Opticks package by other user
--------------------------------------------------------------------------

Sticking points:

* geocache, installcache, optixcache 



CPack ? Decided NO
-----------------------------

As not using a monolithic CMake proj this 
aint convenient as would make 
individual tgz for all 20 subproj

::

    [blyth@localhost opticks]$ cat cmake/Modules/OpticksProjectOptions.cmake

    set(CPACK_GENERATOR TGZ)
    include(CPack)


Remove RPATH of installed libs and executables for easier deployment
-----------------------------------------------------------------------

* do not want to manage a second set of libs and executables 
  without the RPATH so remove that globally from installed libs

* first see what CMake installs by default 

hg diff cmake/Modules/OpticksBuildOptions.cmake::

     set(BUILD_SHARED_LIBS ON)
    -set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    +
    +
    +# add the automatically determined parts of the RPATH
    +# which point to directories outside the build tree to the install RPATH
    +# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    +
    +# the RPATH to be used when installing
    +#SET(CMAKE_INSTALL_RPATH "")
    +


Then full rebuild::

   om-clean
   om-conf
   om-install


CMake emits::

    Set runtime path of "/home/blyth/local/opticks/lib/OKG4Test" to ""


This way forces user to manage LD_LIBRARY_PATH : a recipe for problems.


examples/UseOptiX
---------------------

::

    [blyth@localhost UseOptiX]$ UseOptiX
    UseOptiX: error while loading shared libraries: liboptix.so.6.0.0: cannot open shared object file: No such file or directory
    [blyth@localhost UseOptiX]$ 
    [blyth@localhost UseOptiX]$ 
    [blyth@localhost UseOptiX]$ ldd UseOptiX
    ldd: ./UseOptiX: No such file or directory
    [blyth@localhost UseOptiX]$ ldd $(which UseOptiX)
        linux-vdso.so.1 =>  (0x00007ffe6c98f000)
        liboptix.so.6.0.0 => not found
        liboptixu.so.6.0.0 => not found
        liboptix_prime.so.6.0.0 => not found
        libcurand.so.10 => /usr/local/cuda-10.1/lib64/libcurand.so.10 (0x00007fd1d7211000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007fd1d6f0a000)
        libm.so.6 => /lib64/libm.so.6 (0x00007fd1d6c08000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007fd1d69f2000)
        libc.so.6 => /lib64/libc.so.6 (0x00007fd1d6625000)
        librt.so.1 => /lib64/librt.so.1 (0x00007fd1d641d000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007fd1d6201000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007fd1d5ffd000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fd1db272000)
    [blyth@localhost UseOptiX]$ 


::

    [blyth@localhost UseOptiX]$ LD_LIBRARY_PATH=$(opticks-prefix)/lib:$(opticks-prefix)/lib64:$(opticks-prefix)/externals/lib:$(opticks-prefix)/externals/lib64:$(opticks-prefix)/externals/optix/lib64 UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
     RT_FORMAT_FLOAT4 size 16
    [blyth@localhost UseOptiX]$ 



try $ORIGIN in CMAKE_INSTALL_RPATH
-----------------------------------------


::

     09 #[=[
     10 opticks-llp '$ORIGIN/..'
     11 #]=]
     12 set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/optix/lib64")
     13


Was expecting to need to escape the dollar, but apparently not with CMake 3.13.4::

    [blyth@localhost UseOptiX]$ chrpath /home/blyth/local/opticks/lib/UseOptiX
    /home/blyth/local/opticks/lib/UseOptiX: RPATH=$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/optix/lib64
    [blyth@localhost UseOptiX]$ ldd /home/blyth/local/opticks/lib/UseOptiX
        linux-vdso.so.1 =>  (0x00007ffe7e9a9000)
        liboptix.so.6.0.0 => /home/blyth/local/opticks/lib/../externals/optix/lib64/liboptix.so.6.0.0 (0x00007f11998b5000)
        liboptixu.so.6.0.0 => /home/blyth/local/opticks/lib/../externals/optix/lib64/liboptixu.so.6.0.0 (0x00007f1199523000)
        liboptix_prime.so.6.0.0 => /home/blyth/local/opticks/lib/../externals/optix/lib64/liboptix_prime.so.6.0.0 (0x00007f11985be000)
        libcurand.so.10 => /usr/local/cuda-10.1/lib64/libcurand.so.10 (0x00007f119455d000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f1194256000)
        libm.so.6 => /lib64/libm.so.6 (0x00007f1193f54000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f1193d3e000)
        libc.so.6 => /lib64/libc.so.6 (0x00007f1193971000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007f119376d000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f1199b84000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f1193551000)
        librt.so.1 => /lib64/librt.so.1 (0x00007f1193349000)
    [blyth@localhost UseOptiX]$ 


::

    [blyth@localhost opticks]$ objdump -x $(which OpSnapTest)  | grep RPATH
    RPATH                $ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/optix/lib64




Bundle up $LOCAL_BASE/opticks
--------------------------------

::

    [blyth@localhost opticks]$ du -hs $LOCAL_BASE/opticks
    14G	/home/blyth/local/opticks

    python or bash script to select only whats needed at runtime

    * executables
    * libs 
    * PTX
    * resources ?
  

running from the exploded binary tarball in /tmp/tt
------------------------------------------------------

Simply adjust PATH::

    [blyth@localhost opticks]$ which OpSnapTest
    /tmp/tt/lib/OpSnapTest
    [blyth@localhost opticks]$ chrpath $(which OpSnapTest)
    /tmp/tt/lib/OpSnapTest: RPATH=$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/optix/lib64
    [blyth@localhost opticks]$ 


Expecting to have resource problems, but no it just worked.  Because the topdown locations are all compiled in::

    [blyth@localhost issues]$ which OKConfTest
    /tmp/tt/lib/OKConfTest
    [blyth@localhost issues]$ 
    [blyth@localhost issues]$ 
    [blyth@localhost issues]$ OKConfTest
    OKConf::Dump
                         OKConf::CUDAVersionInteger() 10010
                        OKConf::OptiXVersionInteger() 60000
                   OKConf::ComputeCapabilityInteger() 70
                            OKConf::CMAKE_CXX_FLAGS()  -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-comment -Wno-deprecated -Wno-shadow
                            OKConf::OptiXInstallDir() /usr/local/OptiX_600
                       OKConf::Geant4VersionInteger() 1042
                       OKConf::OpticksInstallPrefix() /home/blyth/local/opticks
                       OKConf::ShaderDir()            /home/blyth/local/opticks/gl

     OKConf::Check() 0


Need a way to override the compiled in install prefix ? OR Perhaps just not do that. Either:

* envvar OPTICKS_INSTALL_PREFIX 
* relative to the location of the binary similar to RPATH $ORIGIN/.. 
  but users can put binaries that use Opticks libs anywhere, so 
  needs to be envvar



need to remake all the examples with the new ORIGIN RPATH
------------------------------------------------------------



ldd shows absolute paths : FIXED
---------------------------------------

::

    [blyth@localhost lib]$ ldd OpSnapTest 
        linux-vdso.so.1 =>  (0x00007ffd481c0000)
        libOKOP.so => /home/blyth/local/opticks/lib64/libOKOP.so (0x00007f3ec3a8f000)
        libOptiXRap.so => /home/blyth/local/opticks/lib64/libOptiXRap.so (0x00007f3ec370c000)
        liboptix.so.6.0.0 => /usr/local/OptiX_600/lib64/liboptix.so.6.0.0 (0x00007f3ec343d000)
        liboptixu.so.6.0.0 => /usr/local/OptiX_600/lib64/liboptixu.so.6.0.0 (0x00007f3ec30ab000)
        liboptix_prime.so.6.0.0 => /usr/local/OptiX_600/lib64/liboptix_prime.so.6.0.0 (0x00007f3ec2146000)
        ...


* :google:`CMake build relocatable binary and libraries`


* https://cmake.org/cmake/help/git-stage/prop_tgt/BUILD_RPATH_USE_ORIGIN.html

This property is initialized by the value of the variable CMAKE_BUILD_RPATH_USE_ORIGIN.

On platforms that support runtime paths (RPATH) with the $ORIGIN token, setting
this property to TRUE enables relative paths in the build RPATH for executables
and shared libraries that point to shared libraries in the same build tree.

Normally the build RPATH of a binary contains absolute paths to the directory
of each shared library it links to. The RPATH entries for directories contained
within the build tree can be made relative to enable relocatable builds and to
help achieve reproducible builds by omitting the build directory from the build
environment.

This property has no effect on platforms that do not support the $ORIGIN token
in RPATH, or when the CMAKE_SKIP_RPATH variable is set. The runtime path set
through the BUILD_RPATH target property is also unaffected by this property.
  


* https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling

* https://stackoverflow.com/questions/48312419/cmake-build-executable-with-relative-paths-for-dependencies-relocatable-executa

As you want to have executable and libraries to be relocatable as whole, using $ORIGIN in RPATH could be your choice.


* https://gitlab.kitware.com/cmake/community/wikis/doc/cmake/RPATH-handling#recommendations

  $ORIGIN: On Linux/Solaris, it's probably a very good idea to specify any
  RPATH setting one requires to look up the location of a package's
  private libraries via a relative expression, to not lose the
  capability to provide a fully relocatable package. This is what
  $ORIGIN is for. In CMAKE_INSTALL_RPATH lines, it should have its
  dollar sign escaped with a backslash to have it end up with proper
  syntax in the final executable. See also the CMake and
  $ORIGIN
  discussion. For Mac OS X, there is a similar @rpath, @loader_path and
  @executable_path mechanism. While dependent libraries use @rpath in
  their install name, relocatable executables should use @loader_path and
  @executable_path in their RPATH. For example, you can set
  CMAKE_INSTALL_RPATH to @loader_path, and if an executable depends on
  "@rpath/libbar.dylib", the loader will then search for
  "@loader_path/libbar.dylib", where @rpath was effectively substituted
  with @loader_path.



CMake and $ORIGIN


* https://cmake.org/pipermail/cmake/2008-January/019290.html

James,

The lack of braces was deliberate - the $ORIGIN string is not a
CMake variable but a special token that should be passed to the
linker without any expansion (the Linux linker provides special
handling for rpath components that use $ORIGIN).



I did try $$ and it helps, but not always (see the end of
the original post). The problem is that $ symbols that are
part of the _value_ of the CMake _LINKER_FLAGS variables
are treated using rules that aren't clear at all (at least
to me). On my system, a single $ is all that's needed for
shared library linker flags but $$ is required for exe
linker flags. But on another system the situation is the
opposite (shared libs get $$, exes get $).

For the time being, I'm using the macro below to paper over
the differences (on Linux, at least).

Iker

# =========================================================
MACRO (APPEND_CMAKE_INSTALL_RPATH RPATH_DIRS)
   IF (NOT ${ARGC} EQUAL 1)
     MESSAGE(SEND_ERROR "APPEND_CMAKE_INSTALL_RPATH takes 1 argument")
   ENDIF (NOT ${ARGC} EQUAL 1)
   FOREACH ( RPATH_DIR ${RPATH_DIRS} )
     IF ( NOT ${RPATH_DIR} STREQUAL "" )
        FILE( TO_CMAKE_PATH ${RPATH_DIR} RPATH_DIR )
        STRING( SUBSTRING ${RPATH_DIR} 0 1 RPATH_FIRST_CHAR )
        IF ( NOT ${RPATH_FIRST_CHAR} STREQUAL "/" )
          # relative path; CMake handling for these is unclear,
          # add them directly to the linker line. Add both $ORIGIN
          # and $$ORIGIN to ensure correct behavior for exes and
          # shared libraries.
          SET ( RPATH_DIR "$ORIGIN/${RPATH_DIR}:$$ORIGIN/${RPATH_DIR}" )
          SET ( CMAKE_EXE_LINKER_FLAGS
                "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,'${RPATH_DIR}'" )
          SET ( CMAKE_SHARED_LINKER_FLAGS
                "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath,'${RPATH_DIR}'" )
        ELSE ( NOT ${RPATH_FIRST_CHAR} STREQUAL "/" )
          # absolute path
          SET ( CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${RPATH_DIR}" )
        ENDIF ( NOT ${RPATH_FIRST_CHAR} STREQUAL "/" )
     ENDIF ( NOT ${RPATH_DIR} STREQUAL "" )
   ENDFOREACH ( RPATH_DIR )
ENDMACRO ( APPEND_CMAKE_INSTALL_RPATH )

The macro takes a list of paths and can be used like this:

    APPEND_CMAKE_INSTALL_RPATH(".;../../;/usr/local/lib")

 > Oh, sorry.  Rereading your mail message more closely, you want a "$"
 > character to pass through properly.
 >
 > Did you try "$$" in the original code (not the one with the single quotes)?
 >
 >     SET(CMAKE_INSTALL_RPATH
 >        "${CMAKE_INSTALL_RPATH}:$$ORIGIN/../xxx")
 >
 > Or perhaps other stuff like on this recent wiki addition?
 >
 > http://www.cmake.org/Wiki/CMake:VariablesListsStrings#Escaping
 >
 > There was a recent thread called "how to escape the $ dollar sign?"
 >
 > James




:google:`RPATH $ORIGIN`


Avoid dollar escaping problems with XORIGIN and chrpath
----------------------------------------------------------

* https://enchildfone.wordpress.com/2010/03/23/a-description-of-rpath-origin-ld_library_path-and-portable-linux-binaries/

$ORIGIN is a special variable that means ‘this executable’, and it means the
actual executable filename, as readlink would see it, so symlinks are followed.
In other words, $ORIGIN is special and resolves to wherever the binary is at
runtime.


So you have to compile the executable so it puts an RPATH in the header.  You
do this by giving a special flag to gcc which will give it to ld, the linker.
It goes like this:

-Wl,-rpath=$ORIGIN/../lib

Getting this value into gcc is not easy.  Because of quoting issues, you can’t
just stick this anywhere, the $ dollar sign gets interpreted by the shell, etc,
so what I like to do is just set it to this:

-Wl,-rpath=XORIGIN/../lib

I replaced the dollar sign with the letter X.  After the binary is compiled and
made I will use chrpath to set the string to what I want it to which is the
same thing with a dollar sign.  Remember the constant pool, that’s why you need
to reserve space in the exe.  This is a trick to side-step the quoting hell
that many people on the net have suffered through, myself included.  Luckily I
saw a neat sidestep.

Coaxing ./configure to get this in there:

LDFLAGS="-Wl,-rpath=XORIGIN/../lib" ./configure --prefix=/blabla/place

See the X? That will be replaced by a dollar sign later when you run chrpath on
the resultant binaries.  The configure script will see the LDFLAGS and pass it
to gcc etc and the build system will incorporate that flag.  See the comma
between -Wl and -rpath?  That’s necessary too.


::

    CHRPATH(1)    change rpath/runpath in binaries    CHRPATH(1)

    NAME
           chrpath - change the rpath or runpath in binaries

    SYNOPSIS
           chrpath [ -v | --version ] [ -d | --delete ] [ -r <path> |  --replace <path> ] 
                   [ -c | --convert ] [ -l | --list ] [ -h | --help ] <program> [ <program> ... ]

    DESCRIPTION
           chrpath  changes,  lists  or  removes  the  rpath or runpath setting in
           a binary.  The rpath, or runpath if it is present, is where the runtime linker
           should look for the libraries needed for a program.

    OPTIONS

           ...

           -r <path> | --replace <path>
                  Replace current rpath or runpath setting with the path given.  
                  The new path must be shorter or the same length as the current path.
           ...

           -l | --list
                  List the current rpath or runpath (default)




LD_TRACE_LOADED_OBJECTS more reliable than ldd
--------------------------------------------------

::

    user@debian:~$ LD_TRACE_LOADED_OBJECTS=1 ./symlinked-ffmpeg
     linux-gate.so.1 =>  (0xb77fc000)
     libavdevice.so.52 => /home/user/i/bin/../lib/libavdevice.so.52 (0xb77f4000)
     libavformat.so.52 => /home/user/i/bin/../lib/libavformat.so.52 (0xb77d9000)
     libavcodec.so.52 => /home/user/i/bin/../lib/libavcodec.so.52 (0xb76d7000)
     libavutil.so.49 => /home/user/i/bin/../lib/libavutil.so.49 (0xb76c6000)
     libm.so.6 => /lib/i686/cmov/libm.so.6 (0xb7692000)
     libc.so.6 => /lib/i686/cmov/libc.so.6 (0xb754b000)
     /lib/ld-linux.so.2 (0xb77fd000)

So this command actually works.  What this command does is set an environment
variable called LD_TRACE_LOADED_OBJECTS and then run the executable.  When the
linux loader sees this env variable has been set, instead of running the exe it
will output the libs that it loads instead and exit.  So you’re seeing the
“real” libs that get loaded rather then some shell script fuckup, which is what
I think ldd is.



Try changing RPATH to find OptiX libs in new location
---------------------------------------------------------

::

    [blyth@localhost lib]$ pwd
    /home/blyth/local/opticks/lib

    [blyth@localhost lib]$ chrpath UseOptiX
    UseOptiX: RPATH=/usr/local/OptiX_600/lib64:/usr/local/cuda-10.1/lib64


::

    [blyth@localhost lib]$ mkdir -p /tmp/tt/lib64/
    [blyth@localhost lib]$ cp -P /usr/local/OptiX_600/lib64/* /tmp/tt/lib64/   ## preserve symbolic links
    [blyth@localhost lib]$ ll /tmp/tt/lib64/
    total 398708
    drwxrwxr-x. 3 blyth blyth        19 Apr 25 21:34 ..
    lrwxrwxrwx. 1 blyth blyth        17 Apr 25 21:34 libcudnn.so.7 -> libcudnn.so.7.3.1
    lrwxrwxrwx. 1 blyth blyth        13 Apr 25 21:34 libcudnn.so -> libcudnn.so.7
    -rwxr-xr-x. 1 blyth blyth 345962592 Apr 25 21:34 libcudnn.so.7.3.1
    lrwxrwxrwx. 1 blyth blyth        26 Apr 25 21:34 liboptix_denoiser.so -> liboptix_denoiser.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        23 Apr 25 21:34 liboptix_prime.so -> liboptix_prime.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth  43365763 Apr 25 21:34 liboptix_denoiser.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth    795949 Apr 25 21:34 liboptix.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        17 Apr 25 21:34 liboptix.so -> liboptix.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth  13958597 Apr 25 21:34 liboptix_prime.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        32 Apr 25 21:34 liboptix_ssim_predictor.so -> liboptix_ssim_predictor.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        18 Apr 25 21:34 liboptixu.so -> liboptixu.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth   2602424 Apr 25 21:34 liboptix_ssim_predictor.so.6.0.0
    drwxrwxr-x. 2 blyth blyth      4096 Apr 25 21:34 .
    -rwxr-xr-x. 1 blyth blyth   1574438 Apr 25 21:34 liboptixu.so.6.0.0
    [blyth@localhost lib]$ 


::

    [blyth@localhost lib]$ chrpath --replace /tmp/tt/lib64:/usr/local/cuda-10.1/lib64 UseOptiX
    UseOptiX: RPATH=/usr/local/OptiX_600/lib64:/usr/local/cuda-10.1/lib64
    UseOptiX: new RPATH: /tmp/tt/lib64:/usr/local/cuda-10.1/lib64
    [blyth@localhost lib]$ 

    [blyth@localhost lib]$ chrpath UseOptiX
    UseOptiX: RPATH=/tmp/tt/lib64:/usr/local/cuda-10.1/lib64


    [blyth@localhost lib]$ UseOptiX   ## still working but is it loading the relocated libs
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
     RT_FORMAT_FLOAT4 size 16
    [blyth@localhost lib]$ 


    [blyth@localhost lib]$ ldd UseOptiX          ## ldd thinks so 
        linux-vdso.so.1 =>  (0x00007ffd37363000)
        liboptix.so.6.0.0 => /tmp/tt/lib64/liboptix.so.6.0.0 (0x00007f867f183000)
        liboptixu.so.6.0.0 => /tmp/tt/lib64/liboptixu.so.6.0.0 (0x00007f867edf1000)
        liboptix_prime.so.6.0.0 => /tmp/tt/lib64/liboptix_prime.so.6.0.0 (0x00007f867de8c000)
        libcurand.so.10 => /usr/local/cuda-10.1/lib64/libcurand.so.10 (0x00007f8679e2b000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f8679b24000)
        libm.so.6 => /lib64/libm.so.6 (0x00007f8679822000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f867960c000)
        libc.so.6 => /lib64/libc.so.6 (0x00007f867923f000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007f867903b000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f867f452000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f8678e1f000)
        librt.so.1 => /lib64/librt.so.1 (0x00007f8678c17000)

    [blyth@localhost lib]$ LD_TRACE_LOADED_OBJECTS=1 ./UseOptiX
        linux-vdso.so.1 =>  (0x00007ffe3d33d000)
        liboptix.so.6.0.0 => /tmp/tt/lib64/liboptix.so.6.0.0 (0x00007fe56e238000)
        liboptixu.so.6.0.0 => /tmp/tt/lib64/liboptixu.so.6.0.0 (0x00007fe56dea6000)
        liboptix_prime.so.6.0.0 => /tmp/tt/lib64/liboptix_prime.so.6.0.0 (0x00007fe56cf41000)
        libcurand.so.10 => /usr/local/cuda-10.1/lib64/libcurand.so.10 (0x00007fe568ee0000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007fe568bd9000)
        libm.so.6 => /lib64/libm.so.6 (0x00007fe5688d7000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007fe5686c1000)
        libc.so.6 => /lib64/libc.so.6 (0x00007fe5682f4000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007fe5680f0000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fe56e507000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007fe567ed4000)
        librt.so.1 => /lib64/librt.so.1 (0x00007fe567ccc000)



::

     find . -name '*.so' ! -path './build/*' ! -path '*.build' 

     find . -name '*.so' ! -path './build/*' ! -path '*\.build*' 




Extracting OptiX with prefix
-------------------------------

::

    [blyth@localhost local]$ pwd
    /usr/local
    [blyth@localhost local]$ sh NVIDIA-OptiX-SDK-6.0.0-linux64-25650775.sh --prefix=/tmp/local

    ...

    Do you accept the license? [yN]: 
    y
    By default the NVIDIA OptiX will be installed in:
      "/tmp/local/NVIDIA-OptiX-SDK-6.0.0-linux64"
    Do you want to include the subdirectory NVIDIA-OptiX-SDK-6.0.0-linux64?
    Saying no will install in: "/tmp/local" [Yn]: 
    n

    Using target directory: /tmp/local
    Extracting, please wait...

    Unpacking finished successfully
    [blyth@localhost local]$ 
    Do you accept the license? [yN]: 
    y
    By default the NVIDIA OptiX will be installed in:
      "/tmp/local/NVIDIA-OptiX-SDK-6.0.0-linux64"
    Do you want to include the subdirectory NVIDIA-OptiX-SDK-6.0.0-linux64?
    Saying no will install in: "/tmp/local" [Yn]: 
    n

    Using target directory: /tmp/local
    Extracting, please wait...

    Unpacking finished successfully
    [blyth@localhost local]$ 


    [blyth@localhost ~]$ ll /tmp/local/
    total 28
    drwxrwxrwt. 23 root  root  8192 Apr 25 22:02 ..
    drwxrwxr-x.  2 blyth blyth 4096 Apr 25 22:03 lib64
    drwxrwxr-x.  2 blyth blyth  221 Apr 25 22:03 doc
    drwxrwxr-x.  5 blyth blyth 4096 Apr 25 22:03 include
    drwxrwxr-x.  4 blyth blyth 4096 Apr 25 22:03 SDK-precompiled-samples
    drwxrwxr-x.  7 blyth blyth   87 Apr 25 22:03 .
    drwxrwxr-x. 41 blyth blyth 4096 Apr 25 22:03 SDK
    [blyth@localhost ~]$ ll /tmp/local/lib64/
    total 398708
    -rwxr-xr-x. 1 blyth blyth 345962592 Jan 26 03:45 libcudnn.so.7.3.1
    -rwxr-xr-x. 1 blyth blyth   2602424 Jan 26 03:56 liboptix_ssim_predictor.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth  43365763 Jan 26 03:56 liboptix_denoiser.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth   1574438 Jan 26 03:56 liboptixu.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth    795949 Jan 26 03:56 liboptix.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth  13958597 Jan 26 03:56 liboptix_prime.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        26 Jan 26 03:57 liboptix_denoiser.so -> liboptix_denoiser.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        13 Jan 26 03:57 libcudnn.so -> libcudnn.so.7
    lrwxrwxrwx. 1 blyth blyth        18 Jan 26 03:57 liboptixu.so -> liboptixu.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        32 Jan 26 03:57 liboptix_ssim_predictor.so -> liboptix_ssim_predictor.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        17 Jan 26 03:57 liboptix.so -> liboptix.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        23 Jan 26 03:57 liboptix_prime.so -> liboptix_prime.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        17 Jan 26 03:57 libcudnn.so.7 -> libcudnn.so.7.3.1
    drwxrwxr-x. 2 blyth blyth      4096 Apr 25 22:03 .
    drwxrwxr-x. 7 blyth blyth        87 Apr 25 22:03 ..
    [blyth@localhost ~]$ 


::

    optix600-install-experimental()
    {
        ## for packaging purposes need to try treating OptiX more like any other external
        cd /usr/local
        local prefix=$LOCAL_BASE/opticks/externals/optix
        mkdir -p $prefix
        echo need to say yes then no to the installer
        sh NVIDIA-OptiX-SDK-6.0.0-linux64-25650775.sh --prefix=$prefix
    }





Try the ORIGIN trick
-----------------------

::

    [blyth@localhost lib]$ chrpath UseOptiX
    UseOptiX: RPATH=/home/blyth/local/opticks/externals/optix/lib64:/usr/local/cuda-10.1/lib64

    [blyth@localhost lib]$ UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
     RT_FORMAT_FLOAT4 size 16


    [blyth@localhost lib]$ pwd
    /home/blyth/local/opticks/lib

    [blyth@localhost lib]$ chrpath -r \$ORIGIN/../externals/optix/lib64:/usr/local/cuda-10.1/lib64 UseOptiX
    UseOptiX: RPATH=/home/blyth/local/opticks/externals/optix/lib64:/usr/local/cuda-10.1/lib64
    UseOptiX: new RPATH: $ORIGIN/../externals/optix/lib64:/usr/local/cuda-10.1/lib64


    [blyth@localhost lib]$ ldd UseOptiX
        linux-vdso.so.1 =>  (0x00007fff71be0000)
        liboptix.so.6.0.0 => /home/blyth/local/opticks/lib/./../externals/optix/lib64/liboptix.so.6.0.0 (0x00007f55eeb56000)
        liboptixu.so.6.0.0 => /home/blyth/local/opticks/lib/./../externals/optix/lib64/liboptixu.so.6.0.0 (0x00007f55ee7c4000)
        liboptix_prime.so.6.0.0 => /home/blyth/local/opticks/lib/./../externals/optix/lib64/liboptix_prime.so.6.0.0 (0x00007f55ed85f000)
        libcurand.so.10 => /usr/local/cuda-10.1/lib64/libcurand.so.10 (0x00007f55e97fe000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f55e94f7000)
        libm.so.6 => /lib64/libm.so.6 (0x00007f55e91f5000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f55e8fdf000)
        libc.so.6 => /lib64/libc.so.6 (0x00007f55e8c12000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007f55e8a0e000)
        /lib64/ld-linux-x86-64.so.2 (0x00007f55eee25000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f55e87f2000)
        librt.so.1 => /lib64/librt.so.1 (0x00007f55e85ea000)
    [blyth@localhost lib]$ l /home/blyth/local/opticks/lib/./../externals/optix/lib64/liboptix.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth 795949 Jan 26 03:56 /home/blyth/local/opticks/lib/./../externals/optix/lib64/liboptix.so.6.0.0
    [blyth@localhost lib]$ 


::

    [blyth@localhost lib]$ LD_TRACE_LOADED_OBJECTS=1 ./UseOptiX
        linux-vdso.so.1 =>  (0x00007fffe6994000)
        liboptix.so.6.0.0 => /home/blyth/local/opticks/lib/../externals/optix/lib64/liboptix.so.6.0.0 (0x00007fe0d7160000)
        liboptixu.so.6.0.0 => /home/blyth/local/opticks/lib/../externals/optix/lib64/liboptixu.so.6.0.0 (0x00007fe0d6dce000)
        liboptix_prime.so.6.0.0 => /home/blyth/local/opticks/lib/../externals/optix/lib64/liboptix_prime.so.6.0.0 (0x00007fe0d5e69000)
        libcurand.so.10 => /usr/local/cuda-10.1/lib64/libcurand.so.10 (0x00007fe0d1e08000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007fe0d1b01000)
        libm.so.6 => /lib64/libm.so.6 (0x00007fe0d17ff000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007fe0d15e9000)
        libc.so.6 => /lib64/libc.so.6 (0x00007fe0d121c000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007fe0d1018000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fe0d742f000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007fe0d0dfc000)
        librt.so.1 => /lib64/librt.so.1 (0x00007fe0d0bf4000)
    [blyth@localhost lib]$ 
    [blyth@localhost lib]$ objdump -x UseOptiX | grep RPATH
      RPATH                $ORIGIN/../externals/optix/lib64:/usr/local/cuda-10.1/lib64
    [blyth@localhost lib]$ 

Create directory structure in /tmp/tt with libs and exe in same relative positions::


    [blyth@localhost tt]$ mkdir -p externals/optix
    [blyth@localhost tt]$ mv lib64 externals/optix/
    [blyth@localhost tt]$ pwd
    /tmp/tt
    [blyth@localhost tt]$ mkdir lib
    [blyth@localhost tt]$ cd lib

Check the ORIGIN RPATH::

    [blyth@localhost lib]$ chrpath UseOptiX 
    UseOptiX: RPATH=$ORIGIN/../externals/optix/lib64:/usr/local/cuda-10.1/lib64
    [blyth@localhost lib]$ l ../externals/optix/lib64/
    total 398704
    -rwxr-xr-x. 1 blyth blyth   1574438 Apr 25 21:34 liboptixu.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth   2602424 Apr 25 21:34 liboptix_ssim_predictor.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        18 Apr 25 21:34 liboptixu.so -> liboptixu.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        32 Apr 25 21:34 liboptix_ssim_predictor.so -> liboptix_ssim_predictor.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth  13958597 Apr 25 21:34 liboptix_prime.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        17 Apr 25 21:34 liboptix.so -> liboptix.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth    795949 Apr 25 21:34 liboptix.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth  43365763 Apr 25 21:34 liboptix_denoiser.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        23 Apr 25 21:34 liboptix_prime.so -> liboptix_prime.so.6.0.0
    lrwxrwxrwx. 1 blyth blyth        26 Apr 25 21:34 liboptix_denoiser.so -> liboptix_denoiser.so.6.0.0
    -rwxr-xr-x. 1 blyth blyth 345962592 Apr 25 21:34 libcudnn.so.7.3.1
    lrwxrwxrwx. 1 blyth blyth        13 Apr 25 21:34 libcudnn.so -> libcudnn.so.7
    lrwxrwxrwx. 1 blyth blyth        17 Apr 25 21:34 libcudnn.so.7 -> libcudnn.so.7.3.1

    [blyth@localhost lib]$ UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
     RT_FORMAT_FLOAT4 size 16
    [blyth@localhost lib]$ 
    [blyth@localhost lib]$ pwd
    /tmp/tt/lib
    [blyth@localhost lib]$ 

    [blyth@localhost lib]$ /tmp/tt/lib/UseOptiX
    OptiX 6.0.0
    Number of Devices = 2

    Device 0: TITAN V
      Compute Support: 7 0
      Total Memory: 12621381632 bytes
    Device 1: TITAN RTX
      Compute Support: 7 5
      Total Memory: 25364987904 bytes
     RT_FORMAT_FLOAT4 size 16
    [blyth@localhost lib]$ 


    [blyth@localhost lib]$ pwd
    /tmp/tt/lib
    [blyth@localhost lib]$ LD_TRACE_LOADED_OBJECTS=1 ./UseOptiX
        linux-vdso.so.1 =>  (0x00007ffc2ab26000)
        liboptix.so.6.0.0 => /tmp/tt/lib/../externals/optix/lib64/liboptix.so.6.0.0 (0x00007fa352e3c000)
        liboptixu.so.6.0.0 => /tmp/tt/lib/../externals/optix/lib64/liboptixu.so.6.0.0 (0x00007fa352aaa000)
        liboptix_prime.so.6.0.0 => /tmp/tt/lib/../externals/optix/lib64/liboptix_prime.so.6.0.0 (0x00007fa351b45000)
        libcurand.so.10 => /usr/local/cuda-10.1/lib64/libcurand.so.10 (0x00007fa34dae4000)
        libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007fa34d7dd000)
        libm.so.6 => /lib64/libm.so.6 (0x00007fa34d4db000)
        libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007fa34d2c5000)
        libc.so.6 => /lib64/libc.so.6 (0x00007fa34cef8000)
        libdl.so.2 => /lib64/libdl.so.2 (0x00007fa34ccf4000)
        /lib64/ld-linux-x86-64.so.2 (0x00007fa35310b000)
        libpthread.so.0 => /lib64/libpthread.so.0 (0x00007fa34cad8000)
        librt.so.1 => /lib64/librt.so.1 (0x00007fa34c8d0000)



RUNPATH vs RPATH
-------------------

* http://longwei.github.io/rpath_origin/

here is the catch, RUNPATH is recommended over RPATH, and RPATH is deprecated,
but RUNPATH is currently not supported by all systems…


* https://software.intel.com/sites/default/files/m/a/1/e/dsohowto.pdf

* ~/opticks_refs/dsohowto.pdf


p40 of 47


For each object, DSO as well as executable, the author can define a “run path”.
The dynamic linker will use the value of the path string when searching for
dependencies of the object the run path is defined in. Run paths comes is two
variants, of which one is deprecated. The runpaths are accessible through
entries in the dynamic section as field with the tags DT_RPATH and DT_RUNPATH.
The difference between the two value is when during the search for
dependencies they are used. The DT_RPATH value is used first, before any other
path, specifically before the path defined in the LD_LIBRARY_PATH environment
variable. This is problematic since it does not allow the user to overwrite
the value. Therefore DT_RPATH is deprecated. The introduction of the new
variant, DT_RUNPATH, corrects this oversight by requiring the value is used
after the path in LD_LIBRARY_PATH.  If both a DT_RPATH and a DT_RUNPATH entry
are available, the former is ignored. To add a string to the run path one
must use the -rpath or -R for the linker. I.e., on the gcc command line one
must use something like gcc -Wl,-rpath,/some/dir:/dir2 file.o

This will add the two named directories to the run path in the order in which
say appear on the command line. If more than one -rpath/-R option is given the
parameters will be concatenated with a separating colon. The order is once
again the same as on the linker command line. For compatibility reasons with
older version of the linker DT RPATH entries are created by default. The linker
op- tion --enable-new-dtags must be used to also add DT RUNPATH entry. This
will cause both, DT RPATH and DT RUNPATH entries, to be created.







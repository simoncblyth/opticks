packaging-opticks-and-externals-for-use-on-gpu-cluster
========================================================

Strategy
----------

1. rearrange the position of libraries such as OptiX to make packaging simpler
2. develop python or bash


Naming the Opticks distribution
--------------------------------

* Name to include versions of gcc and Geant4.
* Not OptiX as will incorporate that in the dist, 
  so its covered by the Opticks version. 


Issue : what to include in binary dist ?  
--------------------------------------------

* executables + libs + PTX  : YES
* external libs 

  * libs without overlap with offline : OptiX, yoctogl, ...   YES INCLUDE
  * libs which offline depends on already  : exclude them and 
    bake the versions of the overlappers into the distro name, 

    * clearly needed for Geant4, ie the version of an Opticks binary dist must have Op
    * but what about boost ? its kinda hidden so maybe not needed 
 
* things needed at runtime : OpticksPhoton.h + gl shaders   
* "public" headers, eg those from G4OK package 
* opticks-config script 

* installcache ?  probably YES
* geocache ? probably NO


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



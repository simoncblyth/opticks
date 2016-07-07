oks-vi(){ vi $(opticks-home)/bin/oks.bash ; }
oks-env(){ echo -n ; }
oks-usage(){  cat << \EOU

Opticks Development Functions
===============================

General usage functions belong in opticks- 
development functions belong here.


oks-i
    edit FindX.cmake for internals     
oks-x
    edit FindX.cmake for xternals     
oks-o
    edit FindX.cmake for others

oks-bash
    edit bash scripts for interal projects 

oks-txt
    edit CMakeLists.txt for interal projects 


issues
---------

D : cold GPU flakey fail
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tried switching OFF in "System Preferences > Energy Saver" the option [Automatic Graphics Switching]
this means graphics always uses the NVIDIA GPU rather that switching, to see if this
changes flakiness.

Unsure about flakiness changes, but that causes GPU memory to fill after a while, operating 
in switching mode seems better at keeping memory available.

Reruning ctests or running them individually does not reproduce the failure::

    The following tests FAILED:
         59 - OptiXRapTest.OScintillatorLibTest (OTHER_FAULT)
         62 - GGeoViewTest.OTracerTest (OTHER_FAULT)

    Application Specific Information:
    abort() called
    *** error for object 0x7fc4630c7c08: incorrect checksum for freed object - object was probably modified after being freed.
     

    Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
    0   libsystem_kernel.dylib          0x00007fff8ea02866 __pthread_kill + 10
    1   libsystem_pthread.dylib         0x00007fff8609f35c pthread_kill + 92
    2   libsystem_c.dylib               0x00007fff8cdefb1a abort + 125
    3   libsystem_malloc.dylib          0x00007fff8681f690 szone_error + 587
    4   libsystem_malloc.dylib          0x00007fff868255f0 small_malloc_from_free_list + 902
    5   libsystem_malloc.dylib          0x00007fff868247b2 szone_malloc_should_clear + 1327
    6   libsystem_malloc.dylib          0x00007fff86826868 malloc_zone_malloc + 71
    7   libsystem_malloc.dylib          0x00007fff8682727c malloc + 42
    8   libc++.1.dylib                  0x00007fff8808928e operator new(unsigned long) + 30
    9   liboptix.1.dylib                0x000000010e82e632 0x10e7dc000 + 337458
    10  liboptix.1.dylib                0x000000010e82f200 0x10e7dc000 + 340480
    11  liboptix.1.dylib                0x000000010eaa1615 0x10e7dc000 + 2905621
    12  liboptix.1.dylib                0x000000010e898884 0x10e7dc000 + 772228
    13  liboptix.1.dylib                0x000000010e7f2001 rtProgramCreateFromPTXFile + 545
    14  libOptiXRap.dylib               0x000000010fb0d67c optix::ContextObj::createProgramFromPTXFile(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) + 620 (optixpp_namespace.h:2166)
    15  libOptiXRap.dylib               0x000000010fb0bb5f OConfig::createProgram(char const*, char const*) + 1999 (OConfig.cc:40)
    16  libOptiXRap.dylib               0x000000010fb12ad0 OContext::createProgram(char const*, char const*) + 48 (OContext.cc:144)
    17  libOptiXRap.dylib               0x000000010fb24f2a OGeo::makeTriangulatedGeometry(GMergedMesh*) + 138 (OGeo.cc:538)
    18  libOptiXRap.dylib               0x000000010fb2327f OGeo::makeGeometry(GMergedMesh*) + 127 (OGeo.cc:429)
    19  libOptiXRap.dylib               0x000000010fb22933 OGeo::convert() + 771 (OGeo.cc:184)
    20  libOpticksOp.dylib              0x000000010fa246d9 OpEngine::prepareOptiX() + 5033 (OpEngine.cc:133)
    21  libGGeoView.dylib               0x000000010f7617d6 App::prepareOptiX() + 326 (App.cc:964)
    22  OTracerTest                     0x000000010d4bb802 main + 994 (OTracerTest.cc:51)
    23  libdyld.dylib                   0x00007fff89e755fd start + 1


This fixes a memory issue, the uploading expecting one scintillator not two.

* https://bitbucket.org/simoncblyth/opticks/commits/6045c1c7b2d9af04ae9213b450d3b03d60d6e171



G5: local boost needs different options ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [  7%] Linking CXX shared library libBoostRap.so
    /usr/bin/ld: /home/blyth/local/opticks/externals/lib/libboost_system.a(error_code.o): relocation R_X86_64_32 against `.rodata.str1.1' can not be used when making a shared object; recompile with -fPIC
    /home/blyth/local/opticks/externals/lib/libboost_system.a: could not read symbols: Bad value
    collect2: ld returned 1 exit status
    gmake[2]: *** [boostrap/libBoostRap.so] Error 1
    gmake[1]: *** [boostrap/CMakeFiles/BoostRap.dir/all] Error 2
    gmake: *** [all] Error 2
    [blyth@ntugrid5 ~]$ 





X(SDU) compute mode to avoid X11 issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [simonblyth@optix ggeoview]$ GGeoViewTest --compute

    2016-07-07 20:11:20.088 INFO  [13832] [OConfig::addProg@80] OConfig::addProg desc OProg R 0 generate.cu.ptx generate  index/raytype 0
    2016-07-07 20:11:20.088 INFO  [13832] [OConfig::addProg@80] OConfig::addProg desc OProg E 0 generate.cu.ptx exception  index/raytype 0
    2016-07-07 20:11:20.088 INFO  [13832] [OPropagator::initRng@161] OPropagator::initRng rng_max 3000000 num_photons 100000 
           rngCacheDir /home/simonblyth/local/env/graphics/ggeoview/cache/rng
    cuRANDWrapper::instanciate with cache enabled : cachedir /home/simonblyth/local/env/graphics/ggeoview/cache/rng
    cuRANDWrapper::fillHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer : no cache /home/simonblyth/local/env/graphics/ggeoview/cache/rng/cuRANDWrapper_3000000_0_0.bin, initing and saving 
    GGeoViewTest: /home/simonblyth/opticks/cudarap/cuRANDWrapper.cc:329: int cuRANDWrapper::LoadIntoHostBuffer(curandState*, unsigned int): Assertion `0' failed.
    Aborted (core dumped)


Compare with D::

    2016-07-07 20:15:40.843 INFO  [14900424] [OConfig::dump@141] OpViz::prepareOptiXVix m_raygen_index 1 m_exception_index 1
    OProg R 0 pinhole_camera.cu.ptx pinhole_camera 
    OProg E 0 pinhole_camera.cu.ptx exception 
    OProg M 1 constantbg.cu.ptx miss 
    2016-07-07 20:15:40.843 INFO  [14900424] [OpEngine::preparePropagator@100] OpEngine::preparePropagator START 
    2016-07-07 20:15:40.843 INFO  [14900424] [OConfig::addProg@80] OConfig::addProg desc OProg R 1 generate.cu.ptx generate  index/raytype 1
    2016-07-07 20:15:40.843 INFO  [14900424] [OConfig::addProg@80] OConfig::addProg desc OProg E 1 generate.cu.ptx exception  index/raytype 1
    2016-07-07 20:15:40.843 INFO  [14900424] [OPropagator::initRng@161] OPropagator::initRng rng_max 3000000 num_photons 100000 rngCacheDir /usr/local/env/graphics/ggeoview/cache/rng
    cuRANDWrapper::instanciate with cache enabled : cachedir /usr/local/env/graphics/ggeoview/cache/rng
    cuRANDWrapper::fillHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer : loading from cache /usr/local/env/graphics/ggeoview/cache/rng/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer 3000000 items from /usr/local/env/graphics/ggeoview/cache/rng/cuRANDWrapper_3000000_0_0.bin load_digest 82f3f46e78f078d98848d3e07ee1a654 


From ggeoview- updating for new name cudarap-::

    ggeoview-rng-max()
    {
       # maximal number of photons that can be handled : move to cudawrap- ?
        echo $(( 1000*1000*3 ))
        #echo $(( 1000*1000*1 ))
    }

    ggeoview-rng-prep()
    {
       cudarap-
       CUDAWRAP_RNG_DIR=$(ggeoview-rng-dir) CUDAWRAP_RNG_MAX=$(ggeoview-rng-max) $(cudarap-ibin)
    }

::

    [simonblyth@optix ~]$ cudarap-
    [simonblyth@optix ~]$ cudarap-ibin
    /home/simonblyth/local/opticks/lib/cuRANDWrapperTest
    [simonblyth@optix ~]$ ll /home/simonblyth/local/opticks/lib/cuRANDWrapperTest
    -rwxr-xr-x. 1 simonblyth simonblyth 493755 Jul  7 19:03 /home/simonblyth/local/opticks/lib/cuRANDWrapperTest


Need to prime the RNG cache::

    [simonblyth@optix cudarap]$ CUDARAP_RNG_DIR=$(ggeoview-rng-dir) CUDARAP_RNG_MAX=$(ggeoview-rng-max) $(cudarap-ibin)
    2016-07-07 20:27:24.710 INFO  [19235] [main@25]  work 3000000 max_blocks 128 seed 0 offset 0 threads_per_block 256 cachedir /home/simonblyth/local/opticks/cache/rng
    cuRANDWrapper::instanciate with cache enabled : cachedir /home/simonblyth/local/opticks/cache/rng
    cuRANDWrapper::Allocate
    cuRANDWrapper::InitFromCacheIfPossible
    cuRANDWrapper::InitFromCacheIfPossible : no cache initing and saving 
    cuRANDWrapper::Init
     init_rng_wrapper sequence_index   0  thread_offset       0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     0.0190 ms 
     init_rng_wrapper sequence_index   1  thread_offset   32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     0.0033 ms 
     init_rng_wrapper sequence_index   2  thread_offset   65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time     0.0027 ms 
    ...


Still looking in wrong place::

    2016-07-07 20:28:44.335 INFO  [20111] [OPropagator::initRng@161] OPropagator::initRng rng_max 3000000 num_photons 100000 rngCacheDir /home/simonblyth/local/env/graphics/ggeoview/cache/rng
    cuRANDWrapper::instanciate with cache enabled : cachedir /home/simonblyth/local/env/graphics/ggeoview/cache/rng
    cuRANDWrapper::fillHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer
    cuRANDWrapper::LoadIntoHostBuffer : no cache /home/simonblyth/local/env/graphics/ggeoview/cache/rng/cuRANDWrapper_3000000_0_0.bin, initing and saving 
    GGeoViewTest: /home/simonblyth/opticks/cudarap/cuRANDWrapper.cc:329: int cuRANDWrapper::LoadIntoHostBuffer(curandState*, unsigned int): Assertion `0' failed.
    Aborted (core dumped)
    [simonblyth@optix cudarap]$ ll /home/simonblyth/local/env/graphics/ggeoview/cache/rng/
    ls: cannot access /home/simonblyth/local/env/graphics/ggeoview/cache/rng/: No such file or directory
    [simonblyth@optix cudarap]$ 
    [simonblyth@optix cudarap]$ ll /home/simonblyth/local/opticks/cache/rng/
    total 129348
    drwxrwxr-x. 3 simonblyth simonblyth        24 Jul  7 20:27 ..
    -rw-rw-r--. 1 simonblyth simonblyth 132000000 Jul  7 20:27 cuRANDWrapper_3000000_0_0.bin
    drwxrwxr-x. 2 simonblyth simonblyth        76 Jul  7 20:27 .
    -rw-rw-r--. 1 simonblyth simonblyth    450560 Jul  7 20:27 cuRANDWrapper_10240_0_0.bin
    [simonblyth@optix cudarap]$ 

::

    (gdb) bt
    #0  0x00007fffefbeb5f7 in raise () from /lib64/libc.so.6
    #1  0x00007fffefbecce8 in abort () from /lib64/libc.so.6
    #2  0x00007fffefbe4566 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffefbe4612 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff37ac86a in cuRANDWrapper::LoadIntoHostBuffer (this=0x3fa1420, host_rng_states=0x7fffcb6ab010, elements=3000000) at /home/simonblyth/opticks/cudarap/cuRANDWrapper.cc:329
    #5  0x00007ffff37acf91 in cuRANDWrapper::fillHostBuffer (this=0x3fa1420, host_rng_states=0x7fffcb6ab010, elements=3000000) at /home/simonblyth/opticks/cudarap/cuRANDWrapper.cc:497
    #6  0x00007ffff1cf2bd1 in OPropagator::initRng (this=0x3fa09e0) at /home/simonblyth/opticks/optixrap/OPropagator.cc:182
    #7  0x00007ffff1cf1501 in OEngineImp::preparePropagator (this=0x1893330) at /home/simonblyth/opticks/optixrap/OEngineImp.cc:167
    #8  0x00007ffff1267f50 in OpEngine::preparePropagator (this=0x18932f0) at /home/simonblyth/opticks/opticksop/OpEngine.cc:101
    #9  0x00007ffff07c2749 in App::preparePropagator (this=0x7fffffffdba0) at /home/simonblyth/opticks/ggeoview/App.cc:981
    #10 0x0000000000403da0 in main (argc=2, argv=0x7fffffffddd8) at /home/simonblyth/opticks/ggeoview/tests/GGeoViewTest.cc:108
    (gdb) 


Config coming form optixrap-/CMakeLists.txt::

    213 set(TARGET    "${name}")
    214 set(PTXDIR    "${CMAKE_INSTALL_PREFIX}/ptx")
    215 set(RNGDIR    "$ENV{LOCAL_BASE}/env/graphics/ggeoview/cache/rng")  # TODO: move into optixrap- OR cudawrap- fiefdom
    216 configure_file(Config.hh.in inc/Config.hh)


After fix that get further::

    simonblyth@optix optixrap]$ GGeoViewTest --compute

    2016-07-07 20:45:07.567 INFO  [23475] [OContext::createIOBuffer@358] OContext::createIOBuffer  (quad)  name               record size (getNumQuads) 2000000 ( 100000,10,2,4)
    2016-07-07 20:45:07.567 INFO  [23475] [OContext::createIOBuffer@330] OContext::createIOBuffer name sequence desc [COMPUTE] (100000,1,2)  NumBytes(0) 1600000 NumBytes(1) 16 NumValues(0) 200000 NumValues(1) 2{}
    2016-07-07 20:45:07.567 INFO  [23475] [OContext::createIOBuffer@346] OContext::createIOBuffer  (USER)  name             sequence size (ijk) 200000 ( 100000,1,2,1) elementsize 8
    OBufBase::getNumBytes RT_FORMAT_USER element_size 8 size 200000 
    2016-07-07 20:45:07.567 INFO  [23475] [OEngineImp::preparePropagator@170] OEngineImp::preparePropagator DONE 
    2016-07-07 20:45:07.567 INFO  [23475] [OpEngine::preparePropagator@102] OpEngine::preparePropagator DONE 
    2016-07-07 20:45:07.567 INFO  [23475] [OpSeeder::seedPhotonsFromGensteps@65] OpSeeder::seedPhotonsFromGensteps
    terminate called after throwing an instance of 'thrust::system::system_error'
      what():  function_attributes(): after cudaFuncGetAttributes: invalid device function
    Aborted (core dumped)
    [simonblyth@optix optixrap]$ 


Maybe lack gensteps::
   
    /// template snow storm /// 
    #24 0x00007ffff17c98bc in unsigned int TBuf::reduce<unsigned int>(unsigned int, unsigned int, unsigned int) const () from /home/simonblyth/local/opticks/lib/libThrustRap.so
    #25 0x00007ffff1260c20 in OpSeeder::seedPhotonsFromGenstepsImp (this=0x3fa35f0, s_gs=..., s_ox=...) at /home/simonblyth/opticks/opticksop/OpSeeder.cc:131
    #26 0x00007ffff1260b31 in OpSeeder::seedPhotonsFromGenstepsViaOptiX (this=0x3fa35f0) at /home/simonblyth/opticks/opticksop/OpSeeder.cc:114
    #27 0x00007ffff1260825 in OpSeeder::seedPhotonsFromGensteps (this=0x3fa35f0) at /home/simonblyth/opticks/opticksop/OpSeeder.cc:72
    #28 0x00007ffff12680c7 in OpEngine::seedPhotonsFromGensteps (this=0x18932e0) at /home/simonblyth/opticks/opticksop/OpEngine.cc:121
    #29 0x00007ffff07c277d in App::seedPhotonsFromGensteps (this=0x7fffffffd960) at /home/simonblyth/opticks/ggeoview/App.cc:987
    #30 0x0000000000403daf in main (argc=2, argv=0x7fffffffdb98) at /home/simonblyth/opticks/ggeoview/tests/GGeoViewTest.cc:110
        (gdb) 
    
    
    
    
X(SDU) test fails
~~~~~~~~~~~~~~~~~~~~
    
    After creating geocache and fixing the reemission texture memory bug are down to two 
test fails from failure to connect to remote X11::


    62/73 Test #62: GGeoViewTest.flagsTest .....................   Passed    0.03 sec
          Start 63: GGeoViewTest.OTracerTest
    63/73 Test #63: GGeoViewTest.OTracerTest ...................***Failed   47.56 sec
          Start 64: GGeoViewTest.GGeoViewTest
    64/73 Test #64: GGeoViewTest.GGeoViewTest ..................***Failed   73.30 sec
          Start 65: GGeoViewTest.LogTest
    65/73 Test #65: GGeoViewTest.LogTest .......................   Passed    0.04 sec
          Start 66: GGeoViewTest.OpEngineTest
    66/73 Test #66: GGeoViewTest.OpEngineTest ..................   Passed    1.19 sec
          Start 67: cfg4Test.CPropLibTest
    67/73 Test #67: cfg4Test.CPropLibTest ......................   Passed    0.06 sec
          Start 68: cfg4Test.CTestDetectorTest
    68/73 Test #68: cfg4Test.CTestDetectorTest .................   Passed    0.09 sec
          Start 69: cfg4Test.CGDMLDetectorTest
    69/73 Test #69: cfg4Test.CGDMLDetectorTest .................   Passed    0.48 sec
          Start 70: cfg4Test.CG4Test
    70/73 Test #70: cfg4Test.CG4Test ...........................   Passed    6.13 sec
          Start 71: cfg4Test.G4MaterialTest
    71/73 Test #71: cfg4Test.G4MaterialTest ....................   Passed    0.08 sec
          Start 72: cfg4Test.G4StringTest
    72/73 Test #72: cfg4Test.G4StringTest ......................   Passed    0.08 sec
          Start 73: cfg4Test.G4BoxTest
    73/73 Test #73: cfg4Test.G4BoxTest .........................   Passed    0.06 sec

    97% tests passed, 2 tests failed out of 73

    Total Test time (real) = 159.35 sec

    The following tests FAILED:
         63 - GGeoViewTest.OTracerTest (Failed)
         64 - GGeoViewTest.GGeoViewTest (Failed)
    Errors while running CTest





X(SDU) test fails
~~~~~~~~~~~~~~~~~~~

::

	The following tests FAILED:
		 50 - ThrustRapTest.TBufTest (Failed)                    ## missing evt 
		 59 - OptiXRapTest.OScintillatorLibTest (OTHER_FAULT)    ## missing buffer, due to no geocache
		 60 - OpticksOpTest.OpIndexerTest (OTHER_FAULT)          ## missing evt  
		 62 - GGeoViewTest.OTracerTest (Failed)
		 63 - GGeoViewTest.GGeoViewTest (Failed)
		 65 - GGeoViewTest.OpEngineTest (OTHER_FAULT)
		 67 - cfg4Test.CTestDetectorTest (SEGFAULT)
		 68 - cfg4Test.CGDMLDetectorTest (OTHER_FAULT)
		 69 - cfg4Test.CG4Test (OTHER_FAULT)


OTracerTest and GGeoViewTest, cannot create OpenGL context remotely::

	2016-07-07 13:35:13.836 INFO  [20533] [App::prepareViz@354] App::prepareViz size 2880,1704,2,0 position 200,200,0,0
	2016-07-07 13:35:13.836 INFO  [20533] [DynamicDefine::write@21] DynamicDefine::write dir /home/simonblyth/local/opticks/gl name dynamic.h
	X11: RandR gamma ramp support seems broken
	GLX: Failed to create context: GLXBadFBConfig[simonblyth@optix ~]$ 

	2016-07-07 13:37:06.814 INFO  [20554] [App::configureViz@340] App::configureViz m_setup bookmarks DONE
	2016-07-07 13:37:06.815 INFO  [20554] [App::prepareViz@354] App::prepareViz size 2880,1704,2,0 position 200,200,0,0
	2016-07-07 13:37:06.815 INFO  [20554] [DynamicDefine::write@21] DynamicDefine::write dir /home/simonblyth/local/opticks/gl name dynamic.h
	X11: RandR gamma ramp support seems broken
	GLX: Failed to create context: GLXBadFBConfig[simonblyth@optix ~]$ 

OpEngineTest::

	2016-07-07 13:38:33.237 INFO  [20567] [OContext::init@126] OContext::init  mode INTEROP num_ray_type 3
	2016-07-07 13:38:33.237 INFO  [20567] [OpEngine::prepareOptiX@112] OpEngine::prepareOptiX (OColors)
	2016-07-07 13:38:33.238 INFO  [20567] [OpEngine::prepareOptiX@118] OpEngine::prepareOptiX (OSourceLib)
	OpEngineTest: /home/simonblyth/opticks/optixrap/OSourceLib.cc:25: void OSourceLib::makeSourceTexture(NPY<float>*): 
        Assertion buf && "OSourceLib::makeSourceTexture NULL buffer, try updating geocache first: ggv -G  ? " failed.
	Aborted (core dumped)


CTestDetectorTest::

	2016-07-07 13:39:50.622 ERROR [20676] [GSurfaceLib::createBufferForTex2d@426] GSurfaceLib::createBufferForTex2d zeros  ni 0 nj 2
	2016-07-07 13:39:50.622 INFO  [20676] [GPropertyLib::close@285] GPropertyLib::close type GSurfaceLib buf NULL

	Program received signal SIGSEGV, Segmentation fault.
	0x00007ffff6908d38 in GPropertyMap<float>::getShortName (this=0x0) at /home/simonblyth/opticks/ggeo/GPropertyMap.cc:237
	237	    return m_shortname ; 
	(gdb) bt
	#0  0x00007ffff6908d38 in GPropertyMap<float>::getShortName() const (this=0x0) at /home/simonblyth/opticks/ggeo/GPropertyMap.cc:237
	#1  0x00007ffff5cf2fa9 in CPropLib::convertMaterial(GMaterial const*) (this=0x68dd80, kmat=0x0) at /home/simonblyth/opticks/cfg4/CPropLib.cc:516
	#2  0x00007ffff5cf0c45 in CPropLib::makeInnerMaterial(char const*) (this=0x68dd80, spec=0x68f638 "Rock/NONE/perfectAbsorbSurface/MineralOil")
	    at /home/simonblyth/opticks/cfg4/CPropLib.cc:205
	#3  0x00007ffff5d1dfc9 in CTestDetector::makeDetector() (this=0x68dcd0) at /home/simonblyth/opticks/cfg4/CTestDetector.cc:125
	#4  0x00007ffff5d1dc34 in CTestDetector::init() (this=0x68dcd0) at /home/simonblyth/opticks/cfg4/CTestDetector.cc:74
	#5  0x00007ffff5d1da6c in CTestDetector::CTestDetector(Opticks*, GGeoTestConfig*, OpticksQuery*) (this=0x68dcd0, cache=0x680920, config=0x68d2b0, query=0x0)
	    at /home/simonblyth/opticks/cfg4/CTestDetector.cc:59
	#6  0x00000000004033bf in main(int, char**) (argc=1, argv=0x7fffffffdda8) at /home/simonblyth/opticks/cfg4/tests/CTestDetectorTest.cc:55
	(gdb) f 1
	#1  0x00007ffff5cf2fa9 in CPropLib::convertMaterial (this=0x68dd80, kmat=0x0) at /home/simonblyth/opticks/cfg4/CPropLib.cc:516
	516	    const char* name = kmat->getShortName();
	(gdb) p kmat
	$1 = (const GMaterial *) 0x0
	(gdb) 

CGDMLDetectorTest::

	2016-07-07 13:47:44.785 INFO  [21100] [CTraverser::Summary@102] CDetector::traverse numMaterials 36 numMaterialsWithoutMPT 36
	2016-07-07 13:47:44.787 WARN  [21100] [CGDMLDetector::addMPT@101] CGDMLDetector::addMPT ALL G4 MATERIALS LACK MPT  FIXING USING G4DAE MATERIALS 
	CGDMLDetectorTest: /home/simonblyth/opticks/cfg4/CGDMLDetector.cc:128: void CGDMLDetector::addMPT(): Assertion `ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material"' failed.
	Aborted (core dumped)



CG4Test, missing G4 env file::

	[simonblyth@optix cfg4]$ CG4Test 
	2016-07-07 13:48:50.186 INFO  [21116] [main@24] CG4Test
	2016-07-07 13:48:50.187 INFO  [21116] [Timer::operator@38] Opticks:: START
	2016-07-07 13:48:50.187 WARN  [21116] [OpticksResource::readG4Environment@321] OpticksResource::readG4Environment MISSING FILE externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 


::

	[simonblyth@optix cfg4]$ g4-ini
	/home/simonblyth/local/opticks/externals/config/geant4.ini
	[simonblyth@optix cfg4]$ g4-export-ini
	=== g4-export-ini : writing G4 environment to /home/simonblyth/local/opticks/externals/config/geant4.ini
	G4LEVELGAMMADATA=/opt/geant4/share/Geant4-10.2.2/data/PhotonEvaporation3.2
	G4NEUTRONXSDATA=/opt/geant4/share/Geant4-10.2.2/data/G4NEUTRONXS1.4
	G4LEDATA=/opt/geant4/share/Geant4-10.2.2/data/G4EMLOW6.48
	G4NEUTRONHPDATA=/opt/geant4/share/Geant4-10.2.2/data/G4NDL4.5
	G4ENSDFSTATEDATA=/opt/geant4/share/Geant4-10.2.2/data/G4ENSDFSTATE1.2.3
	G4RADIOACTIVEDATA=/opt/geant4/share/Geant4-10.2.2/data/RadioactiveDecay4.3.2
	G4ABLADATA=/opt/geant4/share/Geant4-10.2.2/data/G4ABLA3.0
	G4PIIDATA=/opt/geant4/share/Geant4-10.2.2/data/G4PII1.3
	G4SAIDXSDATA=/opt/geant4/share/Geant4-10.2.2/data/G4SAIDDATA1.1
	G4REALSURFACEDATA=/opt/geant4/share/Geant4-10.2.2/data/RealSurface1.0
	[simonblyth@optix cfg4]$ 


After g4-export-ini get to missing material::

	2016-07-07 13:51:44.200 WARN  [21617] [CGDMLDetector::addMPT@101] CGDMLDetector::addMPT ALL G4 MATERIALS LACK MPT  FIXING USING G4DAE MATERIALS 
	2016-07-07 13:51:44.200 INFO  [21617] [GPropertyLib::getIndex@239] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [PPE]
	2016-07-07 13:51:44.200 ERROR [21617] [GMaterialLib::createBufferForTex2d@218] GMaterialLib::createBufferForTex2d NO MATERIALS ?  ni 0 nj 2
	2016-07-07 13:51:44.200 INFO  [21617] [GPropertyLib::close@285] GPropertyLib::close type GMaterialLib buf NULL
	CG4Test: /home/simonblyth/opticks/cfg4/CGDMLDetector.cc:128: void CGDMLDetector::addMPT(): Assertion `ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material"' failed.
	Aborted (core dumped)



Realise that can make geocache without OpenGL context with::

    OpEngineTest --nogeocache

Doing so creates the geocache but fails at::

    2016-07-07 15:35:11.486 INFO  [15392] [OpEngine::prepareOptiX@94] OpEngine::prepareOptiX START
    terminate called after throwing an instance of 'optix::Exception'
      what():  Invalid value
    Aborted (core dumped)
     

After https://bitbucket.org/simoncblyth/opticks/commits/6045c1c7b2d9af04ae9213b450d3b03d60d6e171
that problem is fixed.





X(SDU) xercesc-c dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where are the headers ?

::

	[simonblyth@optix opticks]$ ldd /opt/geant4/lib64/libG4persistency.so  | grep xerces
		libxerces-c-3.1.so => /lib64/libxerces-c-3.1.so (0x00007f8f5cd55000)
	[simonblyth@optix opticks]$ 
	[simonblyth@optix opticks]$ l /lib64/libxerces-c-3.1.so
	-rwxr-xr-x. 1 root root 3853352 Mar 10 23:03 /lib64/libxerces-c-3.1.so
	[simonblyth@optix opticks]$ 


X (SDU) ImGui.so X11 ?
~~~~~~~~~~~~~~~~~~~~~~~~

Below was caused by CMake finding the system static glfw3.a removing name "glfw3" 
enabled to find the desired opticks external .so 

Investigated this with env-;cmak-;cmak-find-GLFW

::

	[ 67%] Built target OGLRap
	Scanning dependencies of target DynamicDefineTest
	[ 67%] Building CXX object oglrap/tests/CMakeFiles/DynamicDefineTest.dir/DynamicDefineTest.cc.o
	[ 67%] Linking CXX executable DynamicDefineTest
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XcursorImageLoadCursor'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XineramaQueryExtension'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XF86VidModeQueryExtension'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRGetCrtcInfo'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRGetCrtcGamma'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XineramaQueryScreens'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XineramaIsActive'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRSelectInput'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XF86VidModeGetGammaRampSize'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XcursorImageDestroy'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XISelectEvents'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRGetOutputPrimary'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XIQueryVersion'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRGetScreenResources'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XF86VidModeGetGammaRamp'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRGetOutputInfo'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRAllocGamma'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRQueryExtension'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRUpdateConfiguration'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRGetScreenResourcesCurrent'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRSetCrtcConfig'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRGetCrtcGammaSize'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XF86VidModeSetGammaRamp'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRSetCrtcGamma'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRFreeOutputInfo'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XcursorImageCreate'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRQueryVersion'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRFreeScreenResources'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRFreeGamma'
	/home/simonblyth/local/opticks/externals/lib/libImGui.so: undefined reference to `XRRFreeCrtcInfo'
	collect2: error: ld returned 1 exit status




SDU : pthreads cmake issue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://github.com/casacore/casacore/issues/104

Even after excluding boost component "thread" cmake still looks for pthread.h and pthread_create::

    -- Boost version: 1.57.0
    -- Found the following Boost libraries:
    --   system
    --   program_options
    --   filesystem
    --   regex
    -- Configuring SysRap
    -- Configuring BoostRap
    -- Configuring NPY
    -- Configuring OpticksCore
    -- Configuring GGeo
    -- Configuring AssimpRap
    -- Configuring OpenMeshRap
    -- Configuring OpticksGeometry
    -- Configuring OGLRap
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Looking for pthread_create
    -- Looking for pthread_create - found
    -- Found Threads: TRUE  






EOU
}


oks-dirs(){  cat << EOL
sysrap
boostrap
opticksnpy
optickscore
ggeo
assimprap
openmeshrap
opticksgeo
oglrap
cudarap
thrustrap
optixrap
opticksop
opticksgl
ggeoview
cfg4
EOL
}

oks-xnames(){ cat << EOX
boost
glm
plog
gleq
glfw
glew
imgui
assimp
openmesh
cuda
thrust
optix
xercesc
g4
zmq
asiozmq
EOX
}

oks-internals(){  cat << EOI
SysRap
BoostRap
NPY
OpticksCore
GGeo
AssimpRap
OpenMeshRap
OpticksGeo
OGLRap
CUDARap
ThrustRap
OptiXRap
OpticksOp
OpticksGL
GGeoView
CfG4
EOI
}
oks-xternals(){  cat << EOX
OpticksBoost
Assimp
OpenMesh
GLM
GLEW
GLEQ
GLFW
ImGui
EnvXercesC
G4DAE
ZMQ
AsioZMQ
EOX
}
oks-other(){  cat << EOO
OpenVR
CNPY
NuWaCLHEP
NuWaGeant4
cJSON
RapSqlite
SQLite3
ChromaPhotonList
G4DAEChroma
NuWaDataModel
ChromaGeant4CLHEP
CLHEP
ROOT
ZMQRoot
EOO
}




oks-tags()
{
   local iwd=$PWD

   opticks-scd

   local hh

   local dir
   oks-dirs | while read dir ; do
       local hh=$(ls -1  $dir/*_API_EXPORT.hh 2>/dev/null)
       local proj=$(dirname $hh)
       local name=$(basename $hh)
       local utag=${name/_API_EXPORT.hh}
       local tag=$(echo $utag | tr "A-Z" "a-z" )
       printf "%20s %30s \n" $proj $utag 
   done

   cd $iwd
}


oks-find-cmake-(){ 
  local f
  local base=$(opticks-home)/CMake/Modules
  local name
  oks-${1} | while read f 
  do
     name=$base/Find${f}.cmake
     [ -f "$name" ] && echo $name
  done 
}

oks-i(){ vi $(oks-find-cmake- internals) ; }
oks-x(){ vi $(oks-find-cmake- xternals) ; }
oks-o(){ vi $(oks-find-cmake- other) ; }

oks-edit(){  opticks-scd ; vi opticks.bash $(oks-bash-list) CMakeLists.txt $(oks-txt-list) ; } 
oks-txt(){   opticks-scd ; vi CMakeLists.txt $(oks-txt-list) ; }
oks-bash(){  opticks-scd ; vi opticks.bash $(oks-bash-list) ; }
oks-tests(){ opticks-scd ; vi $(oks-tests-list) ; } 


oks-name(){ echo Opticks ; }
oks-sln(){ echo $(opticks-bdir)/$(opticks-name).sln ; }
oks-slnw(){  vs- ; echo $(vs-wp $(opticks-sln)) ; }
oks-vs(){ 
   # hmm vs- is from env-
   vs-
   local sln=$1
   [ -z "$sln" ] && sln=$(opticks-sln) 
   local slnw=$(vs-wp $sln)

    cat << EOC
# sln  $sln
# slnw $slnw
# copy/paste into powershell v2 OR just use opticks-vs Powershell function
vs-export 
devenv /useenv $slnw
EOC

}


oks-backup-externals()
{
   # backup externals incase clouds are inaccessible
   # and need to use opticks-nuclear to test complete builds
   cd $LOCAL_BASE
   [ ! -d "opticks_backup" ] && return 
   cp -R opticks/externals opticks_backup/
} 



oks-txt-list(){
  local dir
  oks-dirs | while read dir 
  do
      echo $dir/CMakeLists.txt
  done
}
oks-tests-list(){
  local dir
  local name
  oks-dirs | while read dir 
  do
      name=$dir/tests/CMakeLists.txt
      [ -f "$name" ] && echo $name
  done

}
oks-bash-list(){
  local dir
  local home=$(opticks-home)
  oks-dirs | while read dir 
  do
      ## project folders should have only one .bash excluding any *dev.bash
      local rel=$(ls -1 $home/$dir/*.bash 2>/dev/null | grep -v dev.bash) 

      if [ ! -z "$rel" -a -f "$rel" ]; 
      then
          echo $rel
      else
          echo MISSING $rel
      fi
  done
}
oks-grep()
{
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   opticks-
   local dir 
   local base=$(opticks-home)
   oks-dirs | while read dir 
   do
      local subdirs="${base}/${dir} ${base}/${dir}/tests"
      local sub
      for sub in $subdirs 
      do
         if [ -d "$sub" ]; then 
            cd $sub
            #echo $msg $sub
            grep $* $PWD/*.*
         fi
      done 
   done
   cd $iwd
}


oks-api-export()
{
  opticks-cd 
  local dir
  oks-dirs | while read dir 
  do
      local name=$(ls -1 $dir/*_API_EXPORT.hh 2>/dev/null) 
      [ ! -z "$name" ] && echo $name
  done
}

oks-api-export-vi(){ vi $(oks-api-export) ; }




oks-grep-vi(){ vi $(oks-grep -l ${1:-BLog}) ; }



oks-genrst-(){ local hdr=$1 ; cat << EOR

.. include:: $hdr
   :start-after: /**
   :end-before: **/

EOR
}
oks-genrst()
{
   local hdr=${1:-CGDMLDetector.hh}
   local stem=${hdr/.hh}
   stem=${stem/.hpp} 

   [ ! -f "$hdr" ] && echo $msg header name argument expected to exist hdr $hdr  && return 
   [ "$stem" == "$hdr" ] && echo $msg BAD argument expecting header with name ending .hh or .hpp hdr $hdr stem $stem && return 

   local rst=$stem.rst
   #echo $msg hdr $hdr stem $stem rst $rst 

   [ -f "$rst" ] && echo $msg rst $rst exists already : SKIP && return 

   echo $msg writing sphinx docs shim for hdr $hdr to rst $rst 
   oks-genrst- $hdr > $rst 

}

oks-genrst-auto()
{
   # write the shims for all headers containing "/**"
   local hh=${1:-hh}
   local hdr
   grep -Fl "/**" *.$hh | while read hdr ; do
       oks-genrst $hdr
   done
}

oks-docsrc()
{
   grep -Fl "/**" *.{hh,hpp,cc,cpp} 2>/dev/null | while read hdr ; do
      echo $hdr
   done
}

oks-docvi(){ vi $(oks-docsrc) ; }

########## building sphinx docs

oks-htmldir(){   echo $(opticks-prefix)/html ; }
oks-htmldirbb(){ echo $HOME/simoncblyth.bitbucket.org/opticks ; }
oks-docs()
{
   local iwd=$PWD
   opticks-scd
   local htmldir=$(oks-htmldir)
   local htmldirbb=$(oks-htmldirbb)

   [ -d "$htmldirbb" ] && htmldir=$htmldirbb

   sphinx-build -b html  . $htmldir
   cd $iwd

   open $htmldir/index.html
}

oks-html(){   open $(oks-htmldir)/index.html ; } 
oks-htmlbb(){ open $(oks-htmldirbb)/index.html ; } 








oks-genproj()
{
    # this is typically called from projs like ggeo- 

    local msg=" === $FUNCNAME :"
    local proj=${1}
    local tag=${2}

    [ -z "$proj" -o -z "$tag" ] && echo $msg need both proj $proj and tag $tag  && return 


    importlib-  
    importlib-exports ${proj} ${tag}_API

    plog-
    plog-genlog

    echo $msg merge the below sources into CMakeLists.txt
    oks-genproj-sources- $tag

}



oks-genlog()
{
    opticks-scd 
    local dir
    plog-
    oks-dirs | while read dir 
    do
        opticks-scd $dir

        local name=$(ls -1 *_API_EXPORT.hh 2>/dev/null) 
        [ -z "$name" ] && echo MISSING API_EXPORT in $PWD && return 
        [ ! -z "$name" ] && echo $name

        echo $PWD
        plog-genlog FORCE
    done
}


oks-genproj-sources-(){ 


   local tag=${1:-OKCORE}
   cat << EOS

set(SOURCES
     
    ${tag}_LOG.cc

)
set(HEADERS

    ${tag}_LOG.hh
    ${tag}_API_EXPORT.hh
    ${tag}_HEAD.hh
    ${tag}_TAIL.hh

)
EOS
}


oks-testname(){ echo ${cls}Test.cc ; }
oks-gentest()
{
   local msg=" === $FUNCNAME :"
   local cls=${1:-GMaterial}
   local tag=${2:-GGEO} 

   [ -z "$cls" -o -z "$tag" ] && echo $msg a classname $cls and project tag $tag must be provided && return 
   local name=$(oks-testname $cls)
   [ -f "$name" ] && echo $msg a file named $name exists already in $PWD && return
   echo $msg cls $cls generating test named $name in $PWD
   oks-gentest- $cls $tag > $name
   #cat $name

   vi $name

}
oks-gentest-(){

   local cls=${1:-GMaterial}
   local tag=${2:-GGEO}

   cat << EOT

#include <cassert>
#include "${cls}.hh"

#include "PLOG.hh"
#include "${tag}_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    ${tag}_LOG_ ;




    return 0 ;
}

EOT

}

oks-xcollect-notes(){ cat << EON

*oks-xcollect*
     copies the .bash of externals into externals folder 
     and does inplace edits to correct paths for new home.
     Also writes an externals.bash containing the precursor bash 
     functions.

EON
}
oks-xcollect()
{
   local ehome=$(opticks-home)
   local xhome=$ehome
   local iwd=$PWD 

   cd $ehome

   local xbash=$xhome/externals/externals.bash
   [ ! -d "$xhome/externals" ] && mkdir "$xhome/externals"

   echo "# $FUNCNAME " > $xbash
   
   local x
   local esrc
   local src
   local dst
   local nam
   local note

   oks-xnames | while read x 
   do
      $x-;
      esrc=$($x-source)
      src=${esrc/$ehome\/}
      nam=$(basename $src)
      dst=externals/$nam
       
      if [ -f "$dst" ]; then 
          note="previously copied to dst $dst"  
      else
          note="copying to dst $dst"  
          hg cp $src $xhome/$dst
          perl -pi -e "s,$src,$dst," $xhome/$dst 
          perl -pi -e "s,env-home,opticks-home," $xhome/$dst 
      fi
      printf "# %-15s %15s %35s %s \n" $x $nam $src "$note"
      printf "%-20s %-50s %s\n" "$x-(){" ". \$(opticks-home)/externals/$nam" "&& $x-env \$* ; }"   >> $xbash

   done 
   cd $iwd
}
oks-filemap()
{
   oks-filemap-head
   oks-filemap-body
}

oks-filemap-head(){ cat << EOH
# $FUNCNAME
# configure the spawning of opticks repo from env repo 
# see adm-opticks
#
include opticks.bash
include CMakeLists.txt
include cmake
include externals
#
EOH
}

oks-filemap-body(){
   local dir
   oks-dirs | while read dir ; do
      printf "include %s\n" $dir
   done
}




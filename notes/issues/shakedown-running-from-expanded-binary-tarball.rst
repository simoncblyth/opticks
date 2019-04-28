shakedown-running-from-expanded-binary-tarball
=================================================


Issues
----------

1. BOpticksResource using compiled in install prefix, fixed by 
   allowing OPTICKS_INSTALL_PREFIX envvar to override the OKConf 
   compiled in install prefix

2. Must remember to use "--envkey" when running without opticksdata
   to use the setupViaKey approach

3. OpticksFlags instantiation needs to parse the OpticksPhoton.h header at runtime, 
   fixed by adding just this single header in with the binaries

4. OConfig does some path checking, FIXED

5. now OpSnapTest runs, BUT for real usage on GPU cluster have to 
   strictly divide read-only and writeable locations : I suspect Opticks 
   tends to write liberally : test using another user 



Missing OpticksPhoton.h, FIXED
---------------------------------

::

    opticks-dist-test () 
    { 
        OPTICKS_INSTALL_PREFIX=$(opticks-dist-tmp) \
            gdb --args \
            $(opticks-dist-tmp)/lib/OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=10,eyestartz=-1,eyestopz=5" --size 2560,1440,1 --embedded
    }

    ...


    #4  0x00007ffff3fd057f in OpticksFlags::init (this=0x578e890, path=0x7ffff4028710 "$OPTICKS_INSTALL_PREFIX/include/OpticksCore/OpticksPhoton.h") at /home/blyth/opticks/optickscore/OpticksFlags.cc:411
    #5  0x00007ffff3fd0350 in OpticksFlags::OpticksFlags (this=0x578e890, path=0x7ffff4028710 "$OPTICKS_INSTALL_PREFIX/include/OpticksCore/OpticksPhoton.h") at /home/blyth/opticks/optickscore/OpticksFlags.cc:390
    #6  0x00007ffff3ffad6f in OpticksResource::getFlags (this=0x642f70) at /home/blyth/opticks/optickscore/OpticksResource.cc:1031
    #7  0x00007ffff3ffaefb in OpticksResource::getFlagNames (this=0x642f70) at /home/blyth/opticks/optickscore/OpticksResource.cc:1052
    #8  0x00007ffff3fddfa2 in Opticks::getFlagNames (this=0x7fffffffd710) at /home/blyth/opticks/optickscore/Opticks.cc:2275
    #9  0x00007ffff4bd0169 in GGeo::setupColors (this=0x6491f0) at /home/blyth/opticks/ggeo/GGeo.cc:847
    #10 0x00007ffff4bce35a in GGeo::loadGeometry (this=0x6491f0) at /home/blyth/opticks/ggeo/GGeo.cc:610
    #11 0x00007ffff5fd03b9 in OpticksGeometry::loadGeometryBase (this=0x647af0) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:142
    #12 0x00007ffff5fcfde1 in OpticksGeometry::loadGeometry (this=0x647af0) at /home/blyth/opticks/opticksgeo/OpticksGeometry.cc:92
    #13 0x00007ffff5fd48b0 in OpticksHub::loadGeometry (this=0x63ac50) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:484
    #14 0x00007ffff5fd33b7 in OpticksHub::init (this=0x63ac50) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:224
    #15 0x00007ffff5fd319c in OpticksHub::OpticksHub (this=0x63ac50, ok=0x7fffffffd710) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    #16 0x00007ffff7b56b0d in OpMgr::OpMgr (this=0x7fffffffd6a0, ok=0x7fffffffd710) at /home/blyth/opticks/okop/OpMgr.cc:46
    #17 0x0000000000402cb4 in main (argc=11, argv=0x7fffffffd9a8) at /home/blyth/opticks/okop/tests/OpSnapTest.cc:26

    ...

    #8  0x00007ffff3fddfa2 in Opticks::getFlagNames (this=0x7fffffffd710) at /home/blyth/opticks/optickscore/Opticks.cc:2275
    2275    OpticksAttrSeq* Opticks::getFlagNames() { return m_resource->getFlagNames(); }
    (gdb) f 7
    #7  0x00007ffff3ffaefb in OpticksResource::getFlagNames (this=0x642f70) at /home/blyth/opticks/optickscore/OpticksResource.cc:1052
    warning: Source file is more recent than executable.
    1052        flags->save(dir);
    (gdb) f 6
    #6  0x00007ffff3ffad6f in OpticksResource::getFlags (this=0x642f70) at /home/blyth/opticks/optickscore/OpticksResource.cc:1031
    1031        }
    (gdb) f 5
    #5  0x00007ffff3fd0350 in OpticksFlags::OpticksFlags (this=0x578e890, path=0x7ffff4028710 "$OPTICKS_INSTALL_PREFIX/include/OpticksCore/OpticksPhoton.h") at /home/blyth/opticks/optickscore/OpticksFlags.cc:390
    390     init(path);
    (gdb) 



OConfig path checking  : FIXED by making OKConf sensitive to OPTICKS_INSTALL_PATH too
----------------------------------------------------------------------------------------

::

    2019-04-27 21:21:57.326 INFO  [217538] [OGeo::init@190] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2019-04-27 21:21:57.326 INFO  [217538] [OGeo::convert@214] [ nmm 6
    2019-04-27 21:21:57.326 INFO  [217538] [OGeo::convertMergedMesh@232] ( 0
    2019-04-27 21:21:57.326 FATAL [217538] [OConfig::createProgram@89]  paths do not match  path  /tmp/blyth/opticks/opticks-dist-test/installcache/PTX/OptiXRap_generated_material1_radiance.cu.ptx path2 /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_material1_radiance.cu.ptx
    OpSnapTest: /home/blyth/opticks/optixrap/OConfig.cc:95: optix::Program OConfig::createProgram(const char*, const char*): Assertion `match' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffec066207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffec066207 in raise () from /lib64/libc.so.6
    #1  0x00007fffec0678f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffec05f026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffec05f0d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007ffff77e5dcc in OConfig::createProgram (this=0x5a5d200, cu_name=0x7ffff787e9de "material1_radiance.cu", progname=0x7ffff787e9c9 "closest_hit_radiance") at /home/blyth/opticks/optixrap/OConfig.cc:95
    #5  0x00007ffff77eb8f7 in OContext::createProgram (this=0x5a5d510, cu_filename=0x7ffff787e9de "material1_radiance.cu", progname=0x7ffff787e9c9 "closest_hit_radiance") at /home/blyth/opticks/optixrap/OContext.cc:259
    #6  0x00007ffff7807bf1 in OGeo::makeMaterial (this=0x6df6d20) at /home/blyth/opticks/optixrap/OGeo.cc:419
    #7  0x00007ffff78064ba in OGeo::makeGlobalGeometryGroup (this=0x6df6d20, mm=0x6d1010) at /home/blyth/opticks/optixrap/OGeo.cc:270
    #8  0x00007ffff780615f in OGeo::convertMergedMesh (this=0x6df6d20, i=0) at /home/blyth/opticks/optixrap/OGeo.cc:253
    #9  0x00007ffff7805c81 in OGeo::convert (this=0x6df6d20) at /home/blyth/opticks/optixrap/OGeo.cc:220
    #10 0x00007ffff77fe8e9 in OScene::init (this=0x57db5d0) at /home/blyth/opticks/optixrap/OScene.cc:184
    #11 0x00007ffff77fdae9 in OScene::OScene (this=0x57db5d0, hub=0x63ac60) at /home/blyth/opticks/optixrap/OScene.cc:73
    #12 0x00007ffff7b54dd2 in OpEngine::OpEngine (this=0x57db510, hub=0x63ac60) at /home/blyth/opticks/okop/OpEngine.cc:48
    #13 0x00007ffff7b58996 in OpPropagator::OpPropagator (this=0x57db310, hub=0x63ac60, idx=0x57db2f0) at /home/blyth/opticks/okop/OpPropagator.cc:41
    #14 0x00007ffff7b56bac in OpMgr::OpMgr (this=0x7fffffffd6b0, ok=0x7fffffffd720) at /home/blyth/opticks/okop/OpMgr.cc:51
    #15 0x0000000000402cb4 in main (argc=11, argv=0x7fffffffd9b8) at /home/blyth/opticks/okop/tests/OpSnapTest.cc:26
    (gdb) 
         
::


    2019-04-27 21:21:57.326 FATAL [217538] [OConfig::createProgram@89]  paths do not match  

                path  /tmp/blyth/opticks/opticks-dist-test/installcache/PTX/OptiXRap_generated_material1_radiance.cu.ptx 
               path2             /home/blyth/local/opticks/installcache/PTX/OptiXRap_generated_material1_radiance.cu.ptx


     81 optix::Program OConfig::createProgram(const char* cu_name, const char* progname )
     82 {
     83     std::string path = BOpticksResource::PTXPath(cu_name, m_cmake_target);
     84     std::string path2 = OKConf::PTXPath(  m_cmake_target, cu_name );
     85 
     86     bool match = strcmp(path.c_str(), path2.c_str()) == 0 ;
     87     if(!match)
     88     {
     89         LOG(fatal)
     90              << " paths do not match "
     91              << " path  " << path
     92              << " path2 " << path2
     93              ;
     94     }
     95     assert( match );




Read/Write split running ?
-------------------------------

::

    [blyth@localhost issues]$ sudo -u simon bash
    [simon@localhost issues]$ touch here
    touch: cannot touch ‘here’: Permission denied



/var/tmp/OptixCache permissions problem
--------------------------------------------

* :doc:`var-tmp-OptixCache-permissions-problem`





Rerun from simon gets further::

    [simon@localhost ~]$ OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=10,eyestartz=-1,eyestopz=5" --size 2560,1440,1 --embedded
    2019-04-27 22:45:35.785 INFO  [351375] [BOpticksKey::SetKey@45] from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    2019-04-27 22:45:35.790 ERROR [351375] [OpticksResource::readG4Environment@499]  MISSING inipath /tmp/blyth/opticks/opticks-dist-test/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2019-04-27 22:45:35.790 ERROR [351375] [OpticksResource::readOpticksEnvironment@523]  MISSING inipath /tmp/blyth/opticks/opticks-dist-test/opticksdata/config/opticksdata.ini (create it with bash functions: opticksdata-;opticksdata-export-ini ) 
    2019-04-27 22:45:35.790 ERROR [351375] [OpticksResource::initRunResultsDir@262] /tmp/blyth/opticks/opticks-dist-test/results/OpSnapTest/runlabel/20190427_224535
    2019-04-27 22:45:35.790 INFO  [351375] [OpticksHub::loadGeometry@480] [ /tmp/blyth/opticks/opticks-dist-test/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
    2019-04-27 22:45:36.172 WARN  [351375] [OpticksColors::load@52] OpticksColors::load FAILED no file at  dir /tmp/blyth/opticks/opticks-dist-test/opticksdata/resource/OpticksColors with name OpticksColors.json
    ...
    2019-04-27 22:45:41.590 INFO  [351375] [OGeo::convertMergedMesh@232] ( 5
    2019-04-27 22:45:41.590 INFO  [351375] [OGeo::makeOGeometry@495] ugeocode [T]
    2019-04-27 22:45:41.624 INFO  [351375] [OGeo::convertMergedMesh@264] ) 5 numInstances 480
    2019-04-27 22:45:41.624 INFO  [351375] [OGeo::convert@227] ] nmm 6
    2019-04-27 22:45:41.700 INFO  [351375] [OScene::init@197] ]
    2019-04-27 22:45:41.701 ERROR [351375] [OTracer::init@79]  isTimeTracer NO timetracerscale 1e-06
    2019-04-27 22:45:41.737 INFO  [351375] [OpPropagator::snap@108] OpPropagator::snap
    2019-04-27 22:45:41.737 INFO  [351375] [OpTracer::snap@104] ( BConfig.initial steps=10,eyestartz=-1,eyestopz=5 ekv 3 eki 3 ekf 2 eks 2
    2019-04-27 22:45:41.738 INFO  [351375] [OpticksAim::setTarget@125]  using CenterExtent from m_mesh0  target 352851 aim 1 ce 0.0000,0.0000,19785.0000,1965.0000 for details : --aimdbg
    2019-04-27 22:45:41.826 INFO  [351375] [OTracer::trace_@140] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(2560,1440) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
     i     0 eyez         -1 path /tmp/snap00000.ppm
    Segmentation fault (core dumped)
    [simon@localhost ~]$ 
    [simon@localhost ~]$ 
    [simon@localhost ~]$ 
    [simon@localhost ~]$ ll /tmp/snap00000.ppm
    -rw-rw-r--. 1 blyth blyth 11059217 Apr 27 21:48 /tmp/snap00000.ppm
    [simon@localhost ~]$ 



Still failing after changing to use and resolve $TMP because USER envvar not changed::

    [blyth@localhost sysrap]$ sudo -u simon bash -lc 'echo $USER'
    simon
    [blyth@localhost sysrap]$ sudo -u simon bash -c 'echo $USER'
    blyth

Need the login shell to update USER::

    [blyth@localhost issues]$ sudo -u simon bash -l
    [sudo] password for blyth: 
    [simon@localhost issues]$ echo $USER
    simon






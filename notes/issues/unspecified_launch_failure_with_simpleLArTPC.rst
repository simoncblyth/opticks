unspecified_launch_failure_with_simpleLArTPC
===============================================


Setup
------

0. download test GDML into $HOME/.opticks/GEOM/simpleLArTPC/origin.gdml 
1. use "GEOM" to set envvar to "simpleLArTPC"
2. check conversion with, gxt::

    ./G4CXOpticks_setGeometry_Test.sh


Subsequently find the problem is not specific to that geometry, getting 
it with other geom that has worked recently. 

Likely cause is out of range instanceId. 


Issue
-------


::

    2023-07-28 17:32:52.650 INFO  [380311] [BFile::preparePath@844] created directory /tmp/GEOM/simpleLArTPC/GGeo/GNodeLib
    2023-07-28 17:32:52.653 INFO  [380311] [BFile::preparePath@844] created directory /tmp/GEOM/simpleLArTPC/GGeo/GScintillatorLib/liquidAr
    2023-07-28 17:32:52.655 INFO  [380311] [BFile::preparePath@844] created directory /tmp/GEOM/simpleLArTPC/GGeo/GScintillatorLib/liquidAr_ori
    [New Thread 0x7fffdae9c700 (LWP 380348)]
    [New Thread 0x7fffda518700 (LWP 380349)]
    terminate called after throwing an instance of 'sutil::CUDA_Exception'
      what():  CUDA call (cudaFree( (void*)d_temp_buffer_as ) ) failed with error: 'unspecified launch failure' (/data/blyth/junotop/opticks/CSGOptiX/IAS_Builder.cc:194)


    Program received signal SIGABRT, Aborted.
    0x00007fffeb1ce387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-25.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffeb1ce387 in raise () from /lib64/libc.so.6
    #1  0x00007fffeb1cfa78 in abort () from /lib64/libc.so.6
    #2  0x00007fffebb0bcb3 in __gnu_cxx::__verbose_terminate_handler ()
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/vterminate.cc:95
    #3  0x00007fffebb11e26 in __cxxabiv1::__terminate(void (*)()) ()
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/eh_terminate.cc:47
    #4  0x00007fffebb11e61 in std::terminate ()
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/eh_terminate.cc:57
    #5  0x00007fffebb12094 in __cxxabiv1::__cxa_throw (obj=<optimized out>, tinfo=0x7fffefba1928 <typeinfo for sutil::CUDA_Exception>, dest=
        0x7fffef8748c0 <sutil::CUDA_Exception::~CUDA_Exception()>)
        at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/download/gcc-8.3.0/libstdc++-v3/libsupc++/eh_throw.cc:95
    #6  0x00007fffef8f3ab2 in IAS_Builder::Build (ias=..., instances=...) at /data/blyth/junotop/opticks/CSGOptiX/IAS_Builder.cc:194
    #7  0x00007fffef8f2efa in IAS_Builder::Build (ias=..., ias_inst=..., sbt=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/IAS_Builder.cc:119
    #8  0x00007fffef8f93cf in SBT::createIAS (this=0x215be00, inst=...) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:559
    #9  0x00007fffef8f8989 in SBT::createIAS (this=0x215be00, ias_idx=0) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:346
    #10 0x00007fffef8f8780 in SBT::createIAS_Standard (this=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:318
    #11 0x00007fffef8f8712 in SBT::createIAS (this=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:305
    #12 0x00007fffef8f70df in SBT::createGeom (this=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:112
    #13 0x00007fffef8f6f48 in SBT::setFoundry (this=0x215be00, foundry_=0xef5940) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:92
    #14 0x00007fffef87bd26 in CSGOptiX::initGeometry (this=0x1a77dc0) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:485
    #15 0x00007fffef87ab63 in CSGOptiX::init (this=0x1a77dc0) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:382
    #16 0x00007fffef87a6bf in CSGOptiX::CSGOptiX (this=0x1a77dc0, foundry_=0xef5940) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:352
    #17 0x00007fffef87a199 in CSGOptiX::Create (fd=0xef5940) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:277
    #18 0x00007ffff7b53ad3 in G4CXOpticks::setGeometry_ (this=0x69bb60, fd_=0xef5940) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:360
    #19 0x00007ffff7b5358b in G4CXOpticks::setGeometry (this=0x69bb60, fd_=0xef5940) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:309
    #20 0x00007ffff7b53543 in G4CXOpticks::setGeometry (this=0x69bb60, gg_=0xa10e20) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:272
    #21 0x00007ffff7b5341c in G4CXOpticks::setGeometry (this=0x69bb60, world=0x758850) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:264
    #22 0x00007ffff7b53222 in G4CXOpticks::setGeometry (this=0x69bb60, gdmlpath=0x7fffffffee68 "/home/blyth/.opticks/GEOM/simpleLArTPC/origin.gdml")
        at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:218
    #23 0x00007ffff7b52cef in G4CXOpticks::setGeometry (this=0x69bb60) at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:198
    #24 0x00007ffff7b51c0a in G4CXOpticks::SetGeometry () at /data/blyth/junotop/opticks/g4cx/G4CXOpticks.cc:60
    #25 0x0000000000404d11 in main (argc=1, argv=0x7fffffff5458) at /data/blyth/junotop/opticks/g4cx/tests/G4CXOpticks_setGeometry_Test.cc:16
    (gdb) 



    (gdb) f 17
    #17 0x00007fffef87a199 in CSGOptiX::Create (fd=0xef5940) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:277
    277	    CSGOptiX* cx = new CSGOptiX(fd) ; 
    (gdb) f 16
    #16 0x00007fffef87a6bf in CSGOptiX::CSGOptiX (this=0x1a77dc0, foundry_=0xef5940) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:352
    352	    init(); 
    (gdb) f 15
    #15 0x00007fffef87ab63 in CSGOptiX::init (this=0x1a77dc0) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:382
    382	    initGeometry();
    (gdb) f 14
    #14 0x00007fffef87bd26 in CSGOptiX::initGeometry (this=0x1a77dc0) at /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX.cc:485
    485	    sbt->setFoundry(foundry); 
    (gdb) f 13
    #13 0x00007fffef8f6f48 in SBT::setFoundry (this=0x215be00, foundry_=0xef5940) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:92
    92	    createGeom(); 
    (gdb) f 12
    #12 0x00007fffef8f70df in SBT::createGeom (this=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:112
    112	    createIAS(); 
    (gdb) f 11
    #11 0x00007fffef8f8712 in SBT::createIAS (this=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:305
    305	        createIAS_Standard(); 
    (gdb) f 10
    #10 0x00007fffef8f8780 in SBT::createIAS_Standard (this=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:318
    318	    createIAS(ias_idx); 
    (gdb) p ias_idx
    $1 = 0
    (gdb) f 9
    #9  0x00007fffef8f8989 in SBT::createIAS (this=0x215be00, ias_idx=0) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:346
    346	    createIAS(inst); 
    (gdb) p inst
    $2 = {<std::_Vector_base<qat4, std::allocator<qat4> >> = {
        _M_impl = {<std::allocator<qat4>> = {<__gnu_cxx::new_allocator<qat4>> = {<No data fields>}, <No data fields>}, _M_start = 0x2815240, 
          _M_finish = 0x2815280, _M_end_of_storage = 0x2815280}}, <No data fields>}
    (gdb) p inst.size()
    $3 = 1
    (gdb) f 8
    #8  0x00007fffef8f93cf in SBT::createIAS (this=0x215be00, inst=...) at /data/blyth/junotop/opticks/CSGOptiX/SBT.cc:559
    559	    IAS_Builder::Build(ias, inst, this );
    (gdb) f 7
    #7  0x00007fffef8f2efa in IAS_Builder::Build (ias=..., ias_inst=..., sbt=0x215be00) at /data/blyth/junotop/opticks/CSGOptiX/IAS_Builder.cc:119
    119	    Build(ias, instances); 
    (gdb) 




::

    131 void IAS_Builder::Build(IAS& ias, const std::vector<OptixInstance>& instances)
    132 {
    133     unsigned numInstances = instances.size() ;
    134     LOG(LEVEL) << "numInstances " << numInstances ;
    135 
    136     unsigned numBytes = sizeof( OptixInstance )*numInstances ;
    137 
    138 
    139     CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ias.d_instances ), numBytes ) );
    140     CUDA_CHECK( cudaMemcpy(
    141                 reinterpret_cast<void*>( ias.d_instances ),
    142                 instances.data(),
    143                 numBytes,
    144                 cudaMemcpyHostToDevice
    145                 ) );
    146 
    147 
    148     OptixBuildInput buildInput = {};
    149 
    150     buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    151     buildInput.instanceArray.instances = ias.d_instances ;
    152     buildInput.instanceArray.numInstances = numInstances ;
    153 
    154 
    155     OptixAccelBuildOptions accel_options = {};
    156     accel_options.buildFlags =
    157         OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
    158         OPTIX_BUILD_FLAG_ALLOW_COMPACTION ;
    159     accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;
    160 
    161     OptixAccelBufferSizes as_buffer_sizes;
    162 
    163     OPTIX_CHECK( optixAccelComputeMemoryUsage( Ctx::context, &accel_options, &buildInput, 1, &as_buffer_sizes ) );
    164     CUdeviceptr d_temp_buffer_as;
    165     CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_as ), as_buffer_sizes.tempSizeInBytes ) );




Rerun with::

    export IAS_Builder=INFO
    ./G4CXOpticks_setGeometry_Test.sh


::

    2023-07-28 17:44:32.383 INFO  [380559] [IAS_Builder::Build@109] num_ias_inst 1
    2023-07-28 17:44:32.384 INFO  [380559] [IAS_Builder::Build@114] [ collect OptixInstance 
    2023-07-28 17:44:32.384 INFO  [380559] [IAS_Builder::CollectInstances@70]  i       0 gasIdx   0 sbtOffset      0 gasIdx_sbtOffset.size   1 instanceId 4294967295
    2023-07-28 17:44:32.384 INFO  [380559] [IAS_Builder::Build@116] ] collect OptixInstance 
    2023-07-28 17:44:32.385 INFO  [380559] [IAS_Builder::Build@118] [ build ias 
    2023-07-28 17:44:32.385 INFO  [380559] [IAS_Builder::Build@134] numInstances 1
    terminate called after throwing an instance of 'sutil::CUDA_Exception'
      what():  CUDA call (cudaFree( (void*)d_temp_buffer_as ) ) failed with error: 'unspecified launch failure' (/data/blyth/junotop/opticks/CSGOptiX/IAS_Builder.cc:194)


        Program received signal SIGABRT, Aborted.
    0x00007fffeb1ce387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-25.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) 




Try with geometry that has worked recently::


    G4GDML: Writing setup...
    G4GDML: Writing surfaces...
    G4GDML: Writing '/tmp/GEOM/V1J009/origin_raw.gdml' done !
    2023-07-28 17:49:31.660 INFO  [380679] [U4GDML::write@197]  Apply GDXML::Fix  rawpath /tmp/GEOM/V1J009/origin_raw.gdml dstpath /tmp/GEOM/V1J009/origin.gdml
    2023-07-28 17:49:32.002 ERROR [380679] [GGeo::save_to_dir@785]  default idpath : [/tmp/blyth/opticks/GGeo] is overridden : [/tmp/GEOM/V1J009/GGeo]
    2023-07-28 17:49:32.005 INFO  [380679] [GGeo::save@832]  idpath /tmp/GEOM/V1J009/GGeo
    2023-07-28 17:49:32.037 INFO  [380679] [BFile::preparePath@844] created directory /tmp/GEOM/V1J009/GGeo/GItemList
    2023-07-28 17:49:32.148 INFO  [380679] [BFile::preparePath@844] created directory /tmp/GEOM/V1J009/GGeo/GNodeLib
    2023-07-28 17:49:32.472 INFO  [380679] [BFile::preparePath@844] created directory /tmp/GEOM/V1J009/GGeo/GScintillatorLib/LS
    2023-07-28 17:49:32.474 INFO  [380679] [BFile::preparePath@844] created directory /tmp/GEOM/V1J009/GGeo/GScintillatorLib/LS_ori
    [New Thread 0x7fffd08b9700 (LWP 380724)]
    [New Thread 0x7fffba9dc700 (LWP 380725)]
    2023-07-28 17:49:34.045 INFO  [380679] [IAS_Builder::Build@109] num_ias_inst 48477
    2023-07-28 17:49:34.045 INFO  [380679] [IAS_Builder::Build@114] [ collect OptixInstance 
    2023-07-28 17:49:34.046 INFO  [380679] [IAS_Builder::CollectInstances@70]  i       0 gasIdx   0 sbtOffset      0 gasIdx_sbtOffset.size   1 instanceId 4294967295
    2023-07-28 17:49:34.046 INFO  [380679] [IAS_Builder::CollectInstances@70]  i       1 gasIdx   1 sbtOffset   2977 gasIdx_sbtOffset.size   2 instanceId 4294967295
    2023-07-28 17:49:34.075 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   25601 gasIdx   2 sbtOffset   2982 gasIdx_sbtOffset.size   3 instanceId 4294967295
    2023-07-28 17:49:34.085 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   38216 gasIdx   3 sbtOffset   2990 gasIdx_sbtOffset.size   4 instanceId 4294967295
    2023-07-28 17:49:34.088 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   43213 gasIdx   4 sbtOffset   3001 gasIdx_sbtOffset.size   5 instanceId 4294967295
    2023-07-28 17:49:34.089 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   45613 gasIdx   5 sbtOffset   3006 gasIdx_sbtOffset.size   6 instanceId 4294967295
    2023-07-28 17:49:34.090 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   46203 gasIdx   6 sbtOffset   3007 gasIdx_sbtOffset.size   7 instanceId 4294967295
    2023-07-28 17:49:34.090 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   46793 gasIdx   7 sbtOffset   3008 gasIdx_sbtOffset.size   8 instanceId 4294967295
    2023-07-28 17:49:34.091 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   47383 gasIdx   8 sbtOffset   3009 gasIdx_sbtOffset.size   9 instanceId 4294967295
    2023-07-28 17:49:34.091 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   47973 gasIdx   9 sbtOffset   3010 gasIdx_sbtOffset.size  10 instanceId 4294967295
    2023-07-28 17:49:34.092 INFO  [380679] [IAS_Builder::Build@116] ] collect OptixInstance 
    2023-07-28 17:49:34.092 INFO  [380679] [IAS_Builder::Build@118] [ build ias 
    2023-07-28 17:49:34.092 INFO  [380679] [IAS_Builder::Build@134] numInstances 48477
    terminate called after throwing an instance of 'sutil::CUDA_Exception'
      what():  CUDA call (cudaFree( (void*)d_temp_buffer_as ) ) failed with error: 'unspecified launch failure' (/data/blyth/junotop/opticks/CSGOptiX/IAS_Builder.cc:194)


    Program received signal SIGABRT, Aborted.
    0x00007fffeb1ce387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-4.el7_9.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-59.el7_9.1.x86_64 libicu-50.2-4.el7_7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-25.el7_9.x86_64 openssl-libs-1.0.2k-25.el7_9.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) 


Get the same error.  HMM: the only thing changed with GPU side geometry recently is the sensor id change. 
Maybe that has somehow stomped on instanceId.


Investigate unexpected instanceId
---------------------------------------

::

    2023-07-28 17:49:34.046 INFO  [380679] [IAS_Builder::CollectInstances@70]  i       0 gasIdx   0 sbtOffset      0 gasIdx_sbtOffset.size   1 instanceId 4294967295
    2023-07-28 17:49:34.046 INFO  [380679] [IAS_Builder::CollectInstances@70]  i       1 gasIdx   1 sbtOffset   2977 gasIdx_sbtOffset.size   2 instanceId 4294967295
    2023-07-28 17:49:34.075 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   25601 gasIdx   2 sbtOffset   2982 gasIdx_sbtOffset.size   3 instanceId 4294967295
    2023-07-28 17:49:34.085 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   38216 gasIdx   3 sbtOffset   2990 gasIdx_sbtOffset.size   4 instanceId 4294967295
    2023-07-28 17:49:34.088 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   43213 gasIdx   4 sbtOffset   3001 gasIdx_sbtOffset.size   5 instanceId 4294967295
    2023-07-28 17:49:34.089 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   45613 gasIdx   5 sbtOffset   3006 gasIdx_sbtOffset.size   6 instanceId 4294967295
    2023-07-28 17:49:34.090 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   46203 gasIdx   6 sbtOffset   3007 gasIdx_sbtOffset.size   7 instanceId 4294967295
    2023-07-28 17:49:34.090 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   46793 gasIdx   7 sbtOffset   3008 gasIdx_sbtOffset.size   8 instanceId 4294967295
    2023-07-28 17:49:34.091 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   47383 gasIdx   8 sbtOffset   3009 gasIdx_sbtOffset.size   9 instanceId 4294967295
    2023-07-28 17:49:34.091 INFO  [380679] [IAS_Builder::CollectInstances@70]  i   47973 gasIdx   9 sbtOffset   3010 gasIdx_sbtOffset.size  10 instanceId 4294967295
    2023-07-28 17:49:34.092 INFO  [380679] [IAS_Builder::Build@116] ] collect OptixInstance 


The above instanceId all being the same, and clearly wrong looks like a smoking gun::

    In [1]: np.uint32(-1)                                                                                                                                    
    Out[1]: 4294967295


Start by investigating IAS_Builder::CollectInstances to see where that instanceId is coming from. 

::

     48 void IAS_Builder::CollectInstances(std::vector<OptixInstance>& instances, const std::vector<qat4>& ias_inst, const SBT* sbt ) // static 
     49 {
     50     unsigned num_ias_inst = ias_inst.size() ;
     51     unsigned flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;
     52     unsigned prim_idx = 0u ;  // need sbt offset for the outer prim(aka layer) of the GAS 
     53 
     54     std::map<unsigned, unsigned> gasIdx_sbtOffset ;
     55 
     56     for(unsigned i=0 ; i < num_ias_inst ; i++)
     57     {
     58         const qat4& q = ias_inst[i] ;
     59         int ins_idx,  gasIdx, sensor_identifier, sensor_index ;
     60         q.getIdentity(ins_idx, gasIdx, sensor_identifier, sensor_index );
     61         unsigned instanceId = q.get_IAS_OptixInstance_instanceId() ;
     62 
     63         const GAS& gas = sbt->getGAS(gasIdx);  // susceptible to out-of-range errors for stale gas_idx 
     64 
     65         bool found = gasIdx_sbtOffset.count(gasIdx) == 1 ;
     66         unsigned sbtOffset = found ? gasIdx_sbtOffset.at(gasIdx) : sbt->getOffset(gasIdx, prim_idx ) ;
     67         if(!found)
     68         {
     69             gasIdx_sbtOffset[gasIdx] = sbtOffset ;
     70             LOG(LEVEL)
     71                 << " i " << std::setw(7) << i
     72                 << " gasIdx " << std::setw(3) << gasIdx
     73                 << " sbtOffset " << std::setw(6) << sbtOffset
     74                 << " gasIdx_sbtOffset.size " << std::setw(3) << gasIdx_sbtOffset.size()
     75                 << " instanceId " << instanceId
     76                 ;
     77         }
     78         OptixInstance instance = {} ;
     79         q.copy_columns_3x4( instance.transform );
     80         instance.instanceId = instanceId ;
     81         instance.sbtOffset = sbtOffset ;
     82         instance.visibilityMask = 255;
     83         instance.flags = flags ;
     84         instance.traversableHandle = gas.handle ;
     85    
     86         instances.push_back(instance);
     87     }


Looks like instanceId is exceeding the limit::

     529 /// \see #OptixBuildInputInstanceArray::instances
     530 typedef struct OptixInstance
     531 {
     532     /// affine world-to-object transformation as 3x4 matrix in row-major layout
     533     float transform[12];
     534 
     535     /// Application supplied ID. The maximal ID can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
     536     unsigned int instanceId;
     537 


Rerun with CSGOptiX=INFO dumps the limits::


    2023-07-28 18:04:32.318 INFO  [380998] [CSGOptiX::initCtx@396] 
    Ctx::desc
    Properties::desc
                          limitMaxTraceDepth :         31
               limitMaxTraversableGraphDepth :         16
                    limitMaxPrimitivesPerGas :  536870912  20000000
                     limitMaxInstancesPerIas :   16777216   1000000
                               rtcoreVersion :          0
                          limitMaxInstanceId :   16777215    ffffff
          limitNumBitsInstanceVisibilityMask :          8
                    limitMaxSbtRecordsPerGas :   16777216   1000000
                           limitMaxSbtOffset :   16777215    ffffff





Threaded Properies thru into IAS_Builder::Build to allow checking the 
instanceId is within range. 


HMM::

     530 typedef struct OptixInstance
     531 {
     532     /// affine world-to-object transformation as 3x4 matrix in row-major layout
     533     float transform[12];
     534 
     535     /// Application supplied ID. The maximal ID can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID.
     536     unsigned int instanceId;
     537 
     538     /// SBT record offset.  Will only be used for instances of geometry acceleration structure (GAS) objects.
     539     /// Needs to be set to 0 for instances of instance acceleration structure (IAS) objects. The maximal SBT offset
     540     /// can be queried using OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_SBT_OFFSET.
     541     unsigned int sbtOffset;
     542 
     543     /// Visibility mask. If rayMask & instanceMask == 0 the instance is culled. The number of available bits can be
     544     /// queried using OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK.
     545     unsigned int visibilityMask;
     546 
     547     /// Any combination of OptixInstanceFlags is allowed.
     548     unsigned int flags;
     549 
     550     /// Set with an OptixTraversableHandle.
     551     OptixTraversableHandle traversableHandle;
     552 
     553     /// round up to 80-byte, to ensure 16-byte alignment
     554     unsigned int pad[2];
     555 } OptixInstance;


::

    12 + 1 + 1 + 1 + 1 + 2 + 2 = 20 



OptiX_800_Release_build_on_RTX_5000_Ada_opticks-t_fails
===========================================================

::


    FAILS:  3   / 214   :  Thu Sep  5 21:26:45 2024   
      11 /30  Test #11 : U4Test.U4RandomTest                           ***Failed                      0.03   
      3  /4   Test #3  : CSGOptiXTest.CSGOptiXRenderTest               ***Failed                      3.77   
      1  /2   Test #1  : G4CXTest.G4CXRenderTest                       ***Failed                      5.74   








::

    NP::load Failed to load from path /home/blyth/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy
    /data1/blyth/local/opticks_Release/bin/U4TestRunner.sh : FAIL from U4RandomTest

::

    2024-09-05 21:26:39.206 INFO  [347200] [CSGOptiX::initPIDXYZ@703]  params->pidxyz (4294967295,4294967295,4294967295) 
    2024-09-05 21:26:39.206 INFO  [347200] [G4CXOpticks::setGeometry@259] CSGOptiX::Desc Version 7 WITH_CUSTOM4 
    terminate called after throwing an instance of 'CUDA_Exception'
      what():  CUDA error on synchronize with error 'an illegal memory access was encountered' (/home/blyth/opticks/CSGOptiX/CSGOptiX.cc:1077)

    /data1/blyth/local/opticks_Release/bin/GXTestRunner.sh: line 42: 347200 Aborted                 (core dumped) $EXECUTABLE $@
    /data1/blyth/local/opticks_Release/bin/GXTestRunner.sh : FAIL f



::

    1065 #if OPTIX_VERSION < 70000
    1066     assert( width <= 1000000 );
    1067     six->launch(width, height, depth );
    1068 #else
    1069     if(DEBUG_SKIP_LAUNCH == false)
    1070     {
    1071         CUdeviceptr d_param = (CUdeviceptr)Params::d_param ; ;
    1072         assert( d_param && "must alloc and upload params before launch");
    1073 
    1074         CUstream stream = 0 ;  // default stream 
    1075         OPTIX_CHECK( optixLaunch( pip->pipeline, stream, d_param, sizeof( Params ), &(sbt->sbt), width, height, depth ) );
    1076 
    1077         CUDA_SYNC_CHECK();
    1078         // see CSG/CUDA_CHECK.h the CUDA_SYNC_CHECK does cudaDeviceSyncronize
    1079         // THIS LIKELY HAS LARGE PERFORMANCE IMPLICATIONS : BUT NOT EASY TO AVOID (MULTI-BUFFERING ETC..)  
    1080     }
    1081 #endif




OKTest_CANNOT_createBufferFromGLBO_as_not_uploaded_name_debug
================================================================

::

    OKTest 
    ...
    2020-12-02 15:59:39.465 NONE  [4155154] [OpticksViz::uploadEvent@406] [ (0)
    2020-12-02 15:59:39.487 NONE  [4155154] [OpticksViz::uploadEvent@413] ] (0)
    2020-12-02 15:59:39.570 FATAL [4155154] [OContext::createBuffer@1086] CANNOT createBufferFromGLBO as not uploaded   name                debug buffer_id -1
    Assertion failed: (buffer_id > -1), function createBuffer, file /Users/blyth/opticks/optixrap/OContext.cc, line 1091.
    Abort trap: 6

    (lldb) bt
        frame #3: 0x00007fff685d91ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000100673e9f libOptiXRap.dylib`optix::Handle<optix::BufferObj> OContext::createBuffer<float>(this=0x000000011bf0bfc0, npy=0x0000000122ec6210, name="debug") at OContext.cc:1091
        frame #5: 0x000000010068b877 libOptiXRap.dylib`OEvent::createBuffers(this=0x0000000122faa5f0, evt=0x0000000122d72800) at OEvent.cc:178
        frame #6: 0x000000010068ccbe libOptiXRap.dylib`OEvent::upload(this=0x0000000122faa5f0, evt=0x0000000122d72800) at OEvent.cc:355
        frame #7: 0x000000010068cb1e libOptiXRap.dylib`OEvent::upload(this=0x0000000122faa5f0) at OEvent.cc:344
        frame #8: 0x0000000100598965 libOKOP.dylib`OpEngine::uploadEvent(this=0x00000001119bd3b0) at OpEngine.cc:192
        frame #9: 0x00000001000f3240 libOK.dylib`OKPropagator::uploadEvent(this=0x00000001119cae70) at OKPropagator.cc:131
        frame #10: 0x00000001000f2e56 libOK.dylib`OKPropagator::propagate(this=0x00000001119cae70) at OKPropagator.cc:109
        frame #11: 0x00000001000d5af8 libOK.dylib`OKMgr::propagate(this=0x00007ffeefbfe8b8) at OKMgr.cc:125
        frame #12: 0x000000010000b997 OKTest`main(argc=1, argv=0x00007ffeefbfe978) at OKTest.cc:32
    (lldb) 



Similar recent issue was due to inconsisted interop/compute mode from missing the embedded command line in hasArg.

* :doc:`G4OKTest_CANNOT_createBufferFromGLBO_as_not_uploaded_gensteps.rst`

But this is probably due to the "debug" buffer being new and not having the correct metadata associated.
The "debug" buffer is a purely an output buffer and there is no need to visualize it so can 
make it an OptiX buffer only.  Not OpenGL. 



okc/OpticksBufferSpec::

    const char* OpticksBufferSpec::Get(const char* name, bool compute )

Add "OPTIX_NON_INTEROP" tag to the buffer spec as do not need OpenGL buffers as are not visualizing debug buffers::

    157 const char* OpticksBufferSpec::debug_compute_ = "OPTIX_NON_INTEROP,OPTIX_OUTPUT_ONLY"  ;
    158 const char* OpticksBufferSpec::debug_interop_ = "OPTIX_NON_INTEROP,OPTIX_OUTPUT_ONLY"  ;
    159 



::

    1045 /**
    1046 OContext::createBuffer
    1047 -----------------------
    1048 
    1049 Workhorse, called for example from OEvent::createBuffers
    1050 
    1051 For OpenGL visualized buffers the NPY array must have a valid bufferId 
    1052 indicating that the data was uploaded to an OpenGL buffer by Rdr::upload.
    1053 For buffers that are not visualized such as the "debug" buffer it is 
    1054 necessary for the OpticksBufferSpec/OpticksBufferControl tag of 
    1055 OPTIX_NON_INTEROP to be set to avoid assertions when running interactively.
    1056 See notes/issues/OKTest_CANNOT_createBufferFromGLBO_as_not_uploaded_name_debug.rst 
    1057 
    1058 **/
    1059 
    1060 template <typename T>
    1061 optix::Buffer OContext::createBuffer(NPY<T>* npy, const char* name)
    1062 {   
    1063     assert(npy);
    1064     bool allowed_name = isAllowedBufferName(name); 
    1065     if(!allowed_name) LOG(fatal) << " name " << name << " IS NOT ALLOWED " ;
    1066     assert(allowed_name);
    1067     
    1068     OpticksBufferControl ctrl(npy->getBufferControlPtr());
    1069     bool compute = isCompute()  ;
    1070     
    1071     LOG(LEVEL) 
    1072         << std::setw(20) << name 
    1073         << std::setw(20) << npy->getShapeString() 
    1074         << " mode : " << ( compute ? "COMPUTE " : "INTEROP " )
    1075         << " BufferControl : " << ctrl.description(name)
    1076         ;
    1077     
    1078     unsigned int type(0);
    1079     bool noctrl = false ;
    1080     
    1081     if(      ctrl("OPTIX_INPUT_OUTPUT") )  type = RT_BUFFER_INPUT_OUTPUT ;
    1082     else if( ctrl("OPTIX_OUTPUT_ONLY")  )  type = RT_BUFFER_OUTPUT  ;
    1083     else if( ctrl("OPTIX_INPUT_ONLY")   )  type = RT_BUFFER_INPUT  ;
    1084     else                                   noctrl = true ;
    1085     
    1086     if(noctrl) LOG(fatal) << "no buffer control for " << name << ctrl.description("") ;
    1087     assert(!noctrl);
    1088     
    1089     if( ctrl("BUFFER_COPY_ON_DIRTY") )     type |= RT_BUFFER_COPY_ON_DIRTY ;
    1090     // p170 of OptiX_600 optix-api 
    1091     
    1092     optix::Buffer buffer ;
    1093     
    1094     if( compute )
    1095     {   
    1096         buffer = m_context->createBuffer(type);
    1097     }
    1098     else if( ctrl("OPTIX_NON_INTEROP") )
    1099     {   
    1100         buffer = m_context->createBuffer(type);
    1101     }
    1102     else
    1103     {   
    1104         int buffer_id = npy ? npy->getBufferId() : -1 ;
    1105         if(!(buffer_id > -1))
    1106             LOG(fatal)
    1107                 << "CANNOT createBufferFromGLBO as not uploaded  "
    1108                 << " name " << std::setw(20) << name
    1109                 << " buffer_id " << buffer_id
    1110                 ;
    1111         assert(buffer_id > -1 );
    1112 
    1113         LOG(debug)
    1114             << "createBufferFromGLBO"
    1115             << " name " << std::setw(20) << name
    1116             << " buffer_id " << buffer_id
    1117             ;
    1118 
    1119         buffer = m_context->createBufferFromGLBO(type, buffer_id);
    1120     }
    1121 
    1122     configureBuffer<T>(buffer, npy, name );
    1123     return buffer ;
    1124 }



Gets further, crashing at launch::

    OKTest
    ...

    2020-12-02 17:32:02.799 NONE  [122205] [OpticksViz::uploadEvent@406] [ (0)
    2020-12-02 17:32:02.834 NONE  [122205] [OpticksViz::uploadEvent@413] ] (0)
    2020-12-02 17:32:03.014 INFO  [122205] [OpEngine::close@166]  sensorlib NULL : defaulting it with zero sensors 
    2020-12-02 17:32:03.014 ERROR [122205] [SensorLib::close@362]  SKIP as m_sensor_num zero 
    2020-12-02 17:32:03.038 WARN  [122205] [NPYBase::write_@297]  warning writing empty 
    2020-12-02 17:32:03.064 WARN  [122205] [NPYBase::write_@297]  warning writing empty 
    2020-12-02 17:32:03.064 INFO  [122205] [OpSeeder::seedComputeSeedsFromInteropGensteps@83] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    2020-12-02 17:32:05.548 INFO  [122205] [OPropagator::prelaunch@195] 0 : (0;0,0) 
    2020-12-02 17:32:05.549 INFO  [122205] [BTimes::dump@177] OPropagator::prelaunch
                  validate000                 0.003389
                   compile000                    3e-06
                 prelaunch000                  2.12489
    2020-12-02 17:32:05.549 INFO  [122205] [OPropagator::launch@266] LAUNCH NOW   printLaunchIndex ( -1 -1 -1) -
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuMemcpyDtoHAsync( dstHost, srcDevice, byteCount, hStream.get() ) returned (700): Illegal address)
    Abort trap: 6
    epsilon:opticks blyth$ 


Adding some protection of empty SensorLib avoids that::

    .static __device__ __inline__ float OSensorLib_combined_efficiency(unsigned sensorIndex, float phi_fraction, float theta_fraction  )
     {
         // not expecting sensorIndex 0 which means that the volume is not a sensor
    +
    +    unsigned sensor_data_size = OSensorLib_sensor_data.size(); 
    +    if( sensor_data_size == 0 ) return 1.f ; 
    +
         const float4& sensor_data = OSensorLib_sensor_data[sensorIndex-1] ;  // 1-based sensorIndex
     
         float efficiency_1 = sensor_data.x ; 







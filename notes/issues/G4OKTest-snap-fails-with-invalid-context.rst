
G4OKTest-snap-fails-with-invalid-context
==========================================


Issue : adding snap fails
-----------------------------


::

    @@ -136,6 +136,10 @@ void G4OKTest::init()
         initSensorAngularEfficiency();
         if(m_debug) saveSensorLib(); 
         m_g4ok->uploadSensorLib(); 
    +
    +    const char* dir = "$TMP/g4ok/tests/G4OKTest" ; 
    +    m_g4ok->snap(dir); 
    +
     }


::


    lldb_ G4OKTest 
    ...
    2020-11-06 17:13:24.573 INFO  [1415818] [SensorLib::checkSensorCategories@397] ] SensorLib closed N loaded N sensor_data 672,4 sensor_num 672 sensor_angular_efficiency 1,180,360,1 num_category 1
    2020-11-06 17:13:24.573 INFO  [1415818] [SensorLib::close@359] SensorLib closed Y loaded N sensor_data 672,4 sensor_num 672 sensor_angular_efficiency 1,180,360,1 num_category 1
    2020-11-06 17:13:24.573 INFO  [1415818] [OScene::uploadSensorLib@196] [
    2020-11-06 17:13:24.609 INFO  [1415818] [OSensorLib::makeSensorAngularEfficiencyTexture@106]  item 0 tex_id 4
    2020-11-06 17:13:24.640 INFO  [1415818] [OScene::uploadSensorLib@200] ]
    2020-11-06 17:13:24.640 INFO  [1415818] [OpPropagator::snap@120]  dir $TMP/g4ok/tests/G4OKTest
    2020-11-06 17:13:24.640 INFO  [1415818] [OpTracer::snap@126] ( BConfig.initial  ekv 0 eki 3 ekf 6 eks 2 dir $TMP/g4ok/tests/G4OKTest
    2020-11-06 17:13:24.843 INFO  [1415818] [OTracer::trace_@159]  entry_index 1 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(2880,1704) ZProj.zw (-1.04082,-692676) front 0.7071,0.7071,0.0000
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Variable not found (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Variable "Unresolved reference to variable sequence_buffer from _Z8generatev_cp6" not found in scope)
    Process 10421 stopped
    (lldb) bt
    ...
        frame #8: 0x0000000106445c16 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x0000000117a454c0, code=RT_ERROR_VARIABLE_NOT_FOUND) const at optixpp_namespace.h:1963
        frame #9: 0x0000000106445c77 libOptiXRap.dylib`optix::ContextObj::validate(this=0x0000000117a454c0) at optixpp_namespace.h:2008
        frame #10: 0x0000000106463bec libOptiXRap.dylib`OContext::validate_(this=0x0000000117a43540) at OContext.cc:853
        frame #11: 0x0000000106463606 libOptiXRap.dylib`OContext::launch(this=0x0000000117a43540, lmode=30, entry=1, width=2880, height=1704, times=0x000000011e0745e0) at OContext.cc:816
        frame #12: 0x000000010647d80c libOptiXRap.dylib`OTracer::trace_(this=0x000000011e173880) at OTracer.cc:174
        frame #13: 0x0000000106394b0a libOKOP.dylib`OpTracer::render(this=0x000000011e172f10) at OpTracer.cc:109
        frame #14: 0x0000000106395192 libOKOP.dylib`OpTracer::snap(this=0x000000011e172f10, dir="$TMP/g4ok/tests/G4OKTest") at OpTracer.cc:170
        frame #15: 0x00000001063941ac libOKOP.dylib`OpPropagator::snap(this=0x00000001148dcc80, dir="$TMP/g4ok/tests/G4OKTest") at OpPropagator.cc:121
        frame #16: 0x0000000106393667 libOKOP.dylib`OpMgr::snap(this=0x00000001148d7bf0, dir="$TMP/g4ok/tests/G4OKTest") at OpMgr.cc:179
        frame #17: 0x00000001000e87eb libG4OK.dylib`G4Opticks::snap(this=0x000000010ea5e530, dir="$TMP/g4ok/tests/G4OKTest") const at G4Opticks.cc:630
        frame #18: 0x00000001000110cb G4OKTest`G4OKTest::init(this=0x00007ffeefbfe758) at G4OKTest.cc:141
        frame #19: 0x0000000100010e2f G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe758, argc=1, argv=0x00007ffeefbfe7b8) at G4OKTest.cc:105
        frame #20: 0x0000000100011103 G4OKTest`G4OKTest::G4OKTest(this=0x00007ffeefbfe758, argc=1, argv=0x00007ffeefbfe7b8) at G4OKTest.cc:104
        frame #21: 0x0000000100013bd9 G4OKTest`main(argc=1, argv=0x00007ffeefbfe7b8) at G4OKTest.cc:331
        frame #22: 0x00007fff7c2ec015 libdyld.dylib`start + 1
        frame #23: 0x00007fff7c2ec015 libdyld.dylib`start + 1
    (lldb) 



Crossover between propagate and render ?
---------------------------------------------

The sequence buffer is part of OEvent, hence it should not be needed for rendering.


BUT, surely generate.cu not needed for making a snap::

    epsilon:cu blyth$ grep sequence_buffer *.*
    generate.cu:rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
    generate.cu:        sequence_buffer[photon_id*2 + 0] = seqhis ; 
    generate.cu:        sequence_buffer[photon_id*2 + 1] = seqmat ;  
    epsilon:cu blyth$ 


    epsilon:optixrap blyth$ grep sequence_buffer *.cc
    OContext.cc:    86 rtBuffer<unsigned long long>   sequence_buffer;   // unsigned long long, 8 bytes, 64 bits 
    OEvent.cc:    m_sequence_buffer = m_ocontext->createBuffer<unsigned long long>( sq, "sequence"); 
    OEvent.cc:    m_context["sequence_buffer"]->set( m_sequence_buffer );
    OEvent.cc:    m_sequence_buf = new OBuf("sequence", m_sequence_buffer);
    OEvent.cc:    m_ocontext->resizeBuffer<unsigned long long>(m_sequence_buffer, sq , "sequence");
    OEvent.cc:        OContext::download<unsigned long long>( m_sequence_buffer, sq );
    epsilon:optixrap blyth$ 

::

    127 void OEvent::createBuffers(OpticksEvent* evt)
    128 {



::

    epsilon:opticks blyth$ opticks-c generate.cu
    ./okop/OpSeeder.hh:program cu/generate.cu to access the appropriate values from the genstep buffer
    ./cudarap/cuRANDWrapper.cc:Opticks generate.cu currently does not do the equivalent of this 
    ./opticksgeo/OpticksIdx.cc:    // Indexing the final signed integer boundary code (p.flags.i.x = prd.boundary) from optixrap-/cu/generate.cu
    ./cfg4/CWriter.cc:generate.cu
    ./cfg4/CPhoton.hh:Builds seqhis, seqmat nibble by nibble just like GPU side generate.cu
    ./cfg4/CRecorder.hh:Possibly can drastically simplify (and make much closer to generate.cu) 
    ./ggeo/old/GGeoSensor.cc:in optixrap/cu/generate.cu:generate will find surface properties associated
    ./optickscore/OpticksPhotonFlags.hh:by optixrap/cu/generate.cu
    ./optickscore/OpticksRun.cc:    // that get read into ghead.i.x used in cu/generate.cu
    ./thrustrap/TIsHit.hh:* at each bounce oxrap-/cu/generate.cu FLAGS macro ORs s.flag into p.flags.u.w
    ./thrustrap/iexpand.h:generate.cu program to access the corresponding genstep 
    ./boostrap/BOpticksResource.cc:    const char* name = "generate.cu.ptx" ;
    ./optixrap/OContext.hh:            unsigned int   addEntry(const char* cu_filename="generate.cu", const char* raygen="generate", const char* exception="exception", bool defer=true);
    ./optixrap/OContext.cc:        case 'G': index = addEntry("generate.cu", "generate", "exception", defer) ; break ;
    ./optixrap/OContext.cc:        case 'T': index = addEntry("generate.cu", "trivial",  "exception", defer) ; break ;
    ./optixrap/OContext.cc:        case 'Z': index = addEntry("generate.cu", "zrngtest",  "exception", defer) ; break ;
    ./optixrap/OContext.cc:        case 'N': index = addEntry("generate.cu", "nothing",  "exception", defer) ; break ;
    ./optixrap/OContext.cc:        case 'R': index = addEntry("generate.cu", "tracetest",  "exception", defer) ; break ;
    ./optixrap/OContext.cc:        case 'D': index = addEntry("generate.cu", "dumpseed", "exception", defer) ; break ;
    epsilon:opticks blyth$ 



Both P and G entries are created so it seems need to do something like event uploading 
with empty buffers perhaps for the context to include the expected buffers to be valid  ?

::

     084 OpticksEntry* OContext::addEntry(char code)
      85 {
      86     LOG(LEVEL) << code ;    
      87     
      88     bool defer = true ;
      89     unsigned index ;
      90     switch(code)
      91     {
      92         case 'G': index = addEntry("generate.cu", "generate", "exception", defer) ; break ;
      93         case 'T': index = addEntry("generate.cu", "trivial",  "exception", defer) ; break ;
      94         case 'Z': index = addEntry("generate.cu", "zrngtest",  "exception", defer) ; break ;
      95         case 'N': index = addEntry("generate.cu", "nothing",  "exception", defer) ; break ;
      96         case 'R': index = addEntry("generate.cu", "tracetest",  "exception", defer) ; break ;
      97         case 'D': index = addEntry("generate.cu", "dumpseed", "exception", defer) ; break ;
      98         case 'S': index = addEntry("seedTest.cu", "seedTest", "exception", defer) ; break ;
      99         case 'P': index = addEntry("pinhole_camera.cu", "pinhole_camera" , "exception", defer);  break;
     100     }
     101     return new OpticksEntry(index, code) ;
     102 }

::

    118 void OpPropagator::snap(const char* dir)
    119 {
    120     LOG(info) << " dir " << dir  ;
    121     m_tracer->snap(dir);
    122 }


::

    114 /**
    115 OpTracer::snap
    116 ----------------
    117 
    118 Takes one or more GPU raytrace snapshots of geometry
    119 at various positions configured via --snapconfig
    120 
    121 **/
    122 
    123 void OpTracer::snap(const char* dir)   // --snapconfig="steps=5,eyestartz=0,eyestopz=0"
    124 {
    125 
    126     LOG(info)
    127         << "(" << m_snap_config->desc()
    128         << " dir " << dir
    129         ;
    130 
    131     int num_steps = m_snap_config->steps ;
    132 
    133     float eyestartx = m_snap_config->eyestartx ;
    134     float eyestarty = m_snap_config->eyestarty ;
    135     float eyestartz = m_snap_config->eyestartz ;
    136 
    137     float eyestopx = m_snap_config->eyestopx ;
    138     float eyestopy = m_snap_config->eyestopy ;
    139     float eyestopz = m_snap_config->eyestopz ;
    140 
    141     for(int i=0 ; i < num_steps ; i++)
    142     {   
    ...
    170         render();
    171 
    172         std::cout << " i " << std::setw(5) << i
    173                   << " eyex " << std::setw(10) << eyex
    174                   << " eyey " << std::setw(10) << eyey
    175                   << " eyez " << std::setw(10) << eyez
    176                   << " path " << path
    177                   << std::endl ;
    178 
    179         m_ocontext->snap(path.c_str());
    180     }
    181 
    182     m_otracer->report("OpTracer::snap");   // saves for runresultsdir
    183     //m_ok->dumpMeta("OpTracer::snap");
    184 
    185     m_ok->saveParameters();
    186 
    187     LOG(info) << ")" ;
    188 }


    101 void OpTracer::render()
    102 {
    103     if(m_count == 0 )
    104     {
    105         m_hub->setupCompositionTargetting();
    106         m_otracer->setResolutionScale(1) ;
    107     }
    108 
    109     m_otracer->trace_();
    110     m_count++ ;
    111 }
    112 


    113 void OTracer::trace_()
    114 {
    115     LOG(debug) << "OTracer::trace_ " << m_trace_count ;
    116 
    117     double t0 = BTimeStamp::RealTime();  // THERE IS A HIGHER LEVEL WAY TO DO THIS
    118 
    119     glm::vec3 eye ;
    120     glm::vec3 U ;
    121     glm::vec3 V ;
    122     glm::vec3 W ;
    123     glm::vec4 ZProj ;
    124 
    125     m_composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first
    126 
    127     unsigned cameratype = m_composition->getCameraType();  // 0:PERSP, 1:ORTHO, 2:EQUIRECT
    128     unsigned pixeltime_style = m_composition->getPixelTimeStyle() ;
    129     float    pixeltime_scale = m_composition->getPixelTimeScale() ;
    130     float      scene_epsilon = m_composition->getNear();
    131 
    132     const glm::vec3 front = glm::normalize(W);
    133 
    134     m_context[ "cameratype"]->setUint( cameratype );
    135     m_context[ "pixeltime_style"]->setUint( pixeltime_style );
    136     m_context[ "pixeltime_scale"]->setFloat( pixeltime_scale );
    137     m_context[ "scene_epsilon"]->setFloat(scene_epsilon);
    138     m_context[ "eye"]->setFloat( make_float3( eye.x, eye.y, eye.z ) );
    139     m_context[ "U"  ]->setFloat( make_float3( U.x, U.y, U.z ) );
    140     m_context[ "V"  ]->setFloat( make_float3( V.x, V.y, V.z ) );
    141     m_context[ "W"  ]->setFloat( make_float3( W.x, W.y, W.z ) );
    142     m_context[ "front"  ]->setFloat( make_float3( front.x, front.y, front.z ) );
    143     m_context[ "ZProj"  ]->setFloat( make_float4( ZProj.x, ZProj.y, ZProj.z, ZProj.w ) );
    144 
    145     Buffer buffer = m_context["output_buffer"]->getBuffer();
    146     RTsize buffer_width, buffer_height;
    147     buffer->getSize( buffer_width, buffer_height );
    148 
    ...
    170 
    171     unsigned int lmode = m_trace_count == 0 ? OContext::VALIDATE|OContext::COMPILE|OContext::PRELAUNCH|OContext::LAUNCH : OContext::LAUNCH ;
    172 
    173     //OContext::e_pinhole_camera_entry
    174     m_ocontext->launch( lmode,  m_entry_index,  width, height, m_trace_times );
    175 
    176     double t2 = BTimeStamp::RealTime();
    177 
    178     m_trace_count += 1 ;
    179     m_trace_prep += t1 - t0 ;
    180     m_trace_time += t2 - t1 ;
    181 
    182     //LOG(info) << m_trace_times->description("OTracer::trace m_trace_times") ;
    183 
    184 }




Getting a picture of context population::

    OpMgr=FATAL OGeo=ERROR OContext=ERROR OpEngine=FATAL lldb_ G4OKTest 






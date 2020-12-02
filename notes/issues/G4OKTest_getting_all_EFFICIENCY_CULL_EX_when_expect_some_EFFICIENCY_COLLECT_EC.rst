G4OKTest_getting_all_EFFICIENCY_CULL_EX_when_expect_some_EFFICIENCY_COLLECT_EC
================================================================================


::

    2020-12-02 19:13:06.569 INFO  [185266] [G4OKTest::checkHits@313]  eventID 0 num_hit 47
        0 boundary  -30 sensorIndex   130 nodeIndex  3981 photonIndex    19 flag_mask          5840 sensor_identifier    1005e00 wavelength      430 time  11.5382 SD|BT|TO|EX
        1 boundary  -30 sensorIndex   139 nodeIndex  4035 photonIndex    74 flag_mask          5840 sensor_identifier    1006300 wavelength      430 time   11.576 SD|BT|TO|EX
        2 boundary  -30 sensorIndex   188 nodeIndex  4329 photonIndex   217 flag_mask          5840 sensor_identifier    1008800 wavelength      430 time  13.1592 SD|BT|TO|EX
        3 boundary  -30 sensorIndex    76 nodeIndex  3657 photonIndex   406 flag_mask          5850 sensor_identifier    1003500 wavelength  417.789 time  26.8658 RE|SD|BT|TO|EX
        4 boundary  -30 sensorIndex   189 nodeIndex  4335 photonIndex   546 flag_mask          5840 sensor_identifier    1008900 wavelength      430 time  13.2362 SD|BT|TO|EX
        5 boundary  -30 sensorIndex   181 nodeIndex  4287 photonIndex   586 flag_mask          5c40 sensor_identifier    1008300 wavelength      430 time  15.1188 SD|BR|BT|TO|EX
        6 boundary  -30 sensorIndex   167 nodeIndex  4203 photonIndex   690 flag_mask          5840 sensor_identifier    1007900 wavelength      430 time  12.1973 SD|BT|TO|EX
        7 boundary  -30 sensorIndex   185 nodeIndex  4311 photonIndex   767 flag_mask          5840 sensor_identifier    1008700 wavelength      430 time  13.1909 SD|BT|TO|EX
        8 boundary  -30 sensorIndex   152 nodeIndex  4113 photonIndex  1354 flag_mask          5840 sensor_identifier    1006e00 wavelength      430 time  12.1979 SD|BT|TO|EX
        9 boundary  -30 sensorIndex   150 nodeIndex  4101 photonIndex  1369 flag_mask          5840 sensor_identifier    1006c00 wavelength      430 time  12.2196 SD|BT|TO|EX
       10 boundary  -30 sensorIndex    29 nodeIndex  3375 photonIndex  1406 flag_mask          5840 sensor_identifier    1001000 wavelength      430 time  14.6449 SD|BT|TO|EX


::

    675 #ifdef WITH_ANGULAR
    676     if( s.flag == SURFACE_DETECT )
    677     {
    678         const unsigned& sensorIndex = s.identity.w ;   // should always be > 0 as flag is SD
    679         const float& f_theta =  prd.f_theta ;
    680         const float& f_phi = prd.f_phi ;
    681         const float efficiency = OSensorLib_combined_efficiency(sensorIndex, f_phi, f_theta);
    682         rtPrintf("//SD sensorIndex %d f_theta %f f_phi %f efficiency %f \n", sensorIndex, f_theta, f_phi, efficiency );
    683         float u_angular = curand_uniform(&rng) ;
    684 
    685         p.flags.u.w |= ( u_angular < efficiency ?  EFFICIENCY_COLLECT : EFFICIENCY_CULL ) ;
    686 
    687 #ifdef WITH_DEBUG_BUFFER
    688         debug_buffer[photon_id] = make_float4( f_theta, f_phi, efficiency, unsigned_as_float(sensorIndex) );  
    689 #endif
    690     }
    691 #endif
    692 
    693 #ifdef WITH_DEBUG_BUFFER
    694     //debug_buffer[photon_id] = make_float4( prd.debug.x, prd.debug.y, prd.debug.z, unsigned_as_float(s.identity.y) ); 
    695 #endif
    696 


::

    #!/usr/bin/env python
    """
    ::

        run ~/opticks/ana/debug_buffer.py  

    """
    import os, numpy as np
    np.set_printoptions(suppress=True)

    os.environ.setdefault("OPTICKS_EVENT_BASE",os.path.expandvars("/tmp/$USER/opticks"))
    path = os.path.expandvars("$OPTICKS_EVENT_BASE/G4OKTest/evt/g4live/natural/1/dg.npy")
    dg = np.load(path)

    sensorIndex = dg[:,0,3].view(np.uint32)
    #tid = dg[:,0,3].view(np.uint32)    

    sel = sensorIndex > 0 
    #sel = tid > 0x5000000   # for DYB this means landing (but not necessarily "hitting") a volume of the instanced PMT assembly   

    dgi = sensorIndex[sel]
    dgs = dg[sel]


f_theta coming out nan::

    In [15]: dgs                                                                                                                                                                                              
    Out[15]: 
    array([[[       nan, 0.29379904, 0.        , 0.        ]],

           [[       nan, 0.869334  , 0.        , 0.        ]],

           [[       nan, 0.68843514, 0.        , 0.        ]],

           [[       nan, 0.2709233 , 0.        , 0.        ]],

           [[       nan, 0.14951526, 0.        , 0.        ]],

           [[       nan, 0.02655066, 0.        , 0.        ]],


Add missed normalization::

     52 RT_PROGRAM void closest_hit_propagate()
     53 {
     54      const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     55      float cos_theta = dot(n,ray.direction);
     56 
     57      prd.distance_to_boundary = t ;   // standard semantic attrib for this not available in raygen, so must pass it
     58 
     59      unsigned boundaryIndex = ( instanceIdentity.z & 0xffff ) ;
     60      prd.boundary = cos_theta < 0.f ? -(boundaryIndex + 1) : boundaryIndex + 1 ;
     61      prd.identity = instanceIdentity ;
     62      prd.surface_normal = cos_theta > 0.f ? -n : n ;
     63 
     64      // for angular efficiency 
     65      const float3 isect = ray.origin + t*ray.direction ;
     66      //const float3 local_point = rtTransformPoint( RT_WORLD_TO_OBJECT, isect ); 
     67      const float3 local_point_norm = normalize(rtTransformPoint( RT_WORLD_TO_OBJECT, isect ));
     68 
     69 #ifdef WITH_DEBUG_BUFFER
     70      //prd.debug = isect ; 
     71      prd.debug = local_point_norm ;
     72 #endif
     73 
     74 #ifdef WITH_ANGULAR
     75      const float f_theta = acos( local_point_norm.z )/M_PIf;                             // polar 0->pi ->  0->1
     76      const float f_phi_ = atan2( local_point_norm.y, local_point_norm.x )/(2.f*M_PIf) ;  // azimuthal 0->2pi ->  0->1
     77      const float f_phi = f_phi_ > 0.f ? f_phi_ : f_phi_ + 1.f ;  //  
     78      prd.f_theta = f_theta ;
     79      prd.f_phi = f_phi ;
     80 #endif
     81 


After that start to get EC (EFFICIENCY_COLLECT) as well as EX (EFFICIENCY_CULL)::

    2020-12-02 19:34:31.148 INFO  [200226] [G4OKTest::checkHits@313]  eventID 0 num_hit 47
        0 boundary  -30 sensorIndex   130 nodeIndex  3981 photonIndex    19 flag_mask          9840 sensor_identifier    1005e00 wavelength      430 time  11.5382 SD|BT|TO|EC
        1 boundary  -30 sensorIndex   139 nodeIndex  4035 photonIndex    74 flag_mask          5840 sensor_identifier    1006300 wavelength      430 time   11.576 SD|BT|TO|EX
        2 boundary  -30 sensorIndex   188 nodeIndex  4329 photonIndex   217 flag_mask          5840 sensor_identifier    1008800 wavelength      430 time  13.1592 SD|BT|TO|EX
        3 boundary  -30 sensorIndex    76 nodeIndex  3657 photonIndex   406 flag_mask          5850 sensor_identifier    1003500 wavelength  417.789 time  26.8658 RE|SD|BT|TO|EX
        4 boundary  -30 sensorIndex   189 nodeIndex  4335 photonIndex   546 flag_mask          5840 sensor_identifier    1008900 wavelength      430 time  13.2362 SD|BT|TO|EX
        5 boundary  -30 sensorIndex   181 nodeIndex  4287 photonIndex   586 flag_mask          5c40 sensor_identifier    1008300 wavelength      430 time  15.1188 SD|BR|BT|TO|EX
        6 boundary  -30 sensorIndex   167 nodeIndex  4203 photonIndex   690 flag_mask          5840 sensor_identifier    1007900 wavelength      430 time  12.1973 SD|BT|TO|EX
        7 boundary  -30 sensorIndex   185 nodeIndex  4311 photonIndex   767 flag_mask          9840 sensor_identifier    1008700 wavelength      430 time  13.1909 SD|BT|TO|EC
        8 boundary  -30 sensorIndex   152 nodeIndex  4113 photonIndex  1354 flag_mask          5840 sensor_identifier    1006e00 wavelength      430 time  12.1979 SD|BT|TO|EX
        9 boundary  -30 sensorIndex   150 nodeIndex  4101 photonIndex  1369 flag_mask          5840 sensor_identifier    1006c00 wavelength      430 time  12.2196 SD|BT|TO|EX

    In [16]: run ~/opticks/ana/debug_buffer.py                                                                                                                                                                
    In [17]: dgs                                                                                                                                                                                              
    Out[17]: 
    array([[[0.17026643, 0.29379904, 0.5       , 0.        ]],

           [[0.10552078, 0.869334  , 0.5       , 0.        ]],

           [[0.19465113, 0.68843514, 0.5       , 0.        ]],

           [[0.2230549 , 0.2709233 , 0.        , 0.        ]],

           [[0.13678266, 0.14951526, 0.        , 0.        ]],

           [[0.27697378, 0.02655067, 0.        , 0.        ]],

           [[0.17908172, 0.17669822, 0.5       , 0.        ]],

           [[0.10272076, 0.15709445, 0.5       , 0.        ]],

           [[0.2606528 , 0.6957663 , 0.        , 0.        ]],


Where is this efficiency coming from ?::

    224 void G4OKTest::initSensorAngularEfficiency()
    225 {    
    226     unsigned num_sensor_cat = 1 ;
    227     unsigned num_theta_steps = 180 ;  // height
    228     unsigned num_phi_steps = 360 ;    // width 
    229 
    230     NPY<float>* tab = MockSensorAngularEfficiencyTable::Make(num_sensor_cat, num_theta_steps, num_phi_steps);
    231     m_g4ok->setSensorAngularEfficiency( tab ); 
    232 }


    039 float MockSensorAngularEfficiencyTable::getEfficiency(unsigned i_cat, unsigned j_theta, unsigned k_phi) const
     40 {
     41     float theta =  m_theta_min + j_theta*m_theta_step ;
     42     float phi = m_phi_min + k_phi*m_phi_step ;
     43     const float twopi = 2.f*glm::pi<float>() ;
     44 
     45     float phi_eff = i_cat == 0 ? 1.f : cos(phi*twopi/360.f) ;           // some variation in phi 
     46     float theta_eff = int(theta/10.) % 2 == 0 ? 0.f : 1.f ;  // stripped test function
     47 
     48     return phi_eff*theta_eff ;
     49 }


::

    run ~/opticks/ana/SensorLib.py 


Artificial angular banding flipping from 0. to 1.



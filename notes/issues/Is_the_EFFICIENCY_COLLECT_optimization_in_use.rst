Is_the_EFFICIENCY_COLLECT_optimization_in_use
================================================




Overview
---------

In old Opticks had memory optimization that did efficiency culling
on GPU using the below EFFICIENCY enum, such that only collected hits
needed CPU memory allocation.  

sysrap/OpticksPhoton.h::

     37     NAN_ABORT         = 0x1 << 13,
     38     EFFICIENCY_CULL    = 0x1 << 14,
     39     EFFICIENCY_COLLECT = 0x1 << 15,
     40     __NATURAL         = 0x1 << 16,
     41     __MACHINERY       = 0x1 << 17,

   
TODO: check Custom4 


Review
--------



::

    epsilon:junosw blyth$ opticks-f EFFICIENCY_

    ./sysrap/OpticksPhoton.cc:        "EFFICIENCY_COLLECT":"pink",
    ./sysrap/OpticksPhoton.cc:        "EFFICIENCY_CULL":"red"
    ./sysrap/tests/sphoton_test.cc:       |= EFFICIENCY_CULL 00000000000000000111111111111111 15
    ./sysrap/tests/sphoton_test.cc:    |= EFFICIENCY_COLLECT 00000000000000001111111111111111 16
    ./sysrap/tests/sphoton_test.cc:    flagmask |= EFFICIENCY_CULL ;    dump("|= EFFICIENCY_CULL", flagmask ); 
    ./sysrap/tests/sphoton_test.cc:    flagmask |= EFFICIENCY_COLLECT ; dump("|= EFFICIENCY_COLLECT", flagmask) ;  
    ./sysrap/OpticksPhoton.hh:    static constexpr const char* EFFICIENCY_CULL_ = "EFFICIENCY_CULL" ;
    ./sysrap/OpticksPhoton.hh:    static constexpr const char* EFFICIENCY_COLLECT_ = "EFFICIENCY_COLLECT" ;
    ./sysrap/OpticksPhoton.hh:    static constexpr const char* _EFFICIENCY_COLLECT = "EC" ;
    ./sysrap/OpticksPhoton.hh:    static constexpr const char* _EFFICIENCY_CULL    = "EX" ;
    ./sysrap/OpticksPhoton.hh:        case EFFICIENCY_CULL:    s=EFFICIENCY_CULL_ ;break; 
    ./sysrap/OpticksPhoton.hh:        case EFFICIENCY_COLLECT: s=EFFICIENCY_COLLECT_ ;break; 
    ./sysrap/OpticksPhoton.hh:        case EFFICIENCY_COLLECT: s=_EFFICIENCY_COLLECT ;break; 
    ./sysrap/OpticksPhoton.hh:        case EFFICIENCY_CULL:    s=_EFFICIENCY_CULL ;break; 
    ./sysrap/OpticksPhoton.hh:    pairs.push_back(KV(EFFICIENCY_CULL_ , _EFFICIENCY_CULL)); 
    ./sysrap/OpticksPhoton.hh:    pairs.push_back(KV(EFFICIENCY_COLLECT_ , _EFFICIENCY_COLLECT)); 
    ./sysrap/OpticksPhoton.h:    EFFICIENCY_CULL    = 0x1 << 14,
    ./sysrap/OpticksPhoton.h:    EFFICIENCY_COLLECT = 0x1 << 15,


    ## all the below is dead code...  

    ./cfg4/CCtx.cc:Adds EFFICIENCY_COLLECT or EFFICIENCY_CULL to the _hitflags, 
    ./cfg4/CCtx.cc:        _hitflags |= EFFICIENCY_COLLECT ; 
    ./cfg4/CCtx.cc:        _hitflags |= EFFICIENCY_CULL ; 
    ./g4ok/G4Opticks.cc:    EFFICIENCY_COLLECT (EC) or EFFICIENCY_CULL (EX) flag for photons that already 

    ./optickscore/tests/OpticksFlagsTest.cc:    OpticksFlagsTest --dbghitmask SD,EC           # EC: EFFICIENCY_COLLECT 

    ./examples/Geant4/OpticalApp/OpticalRecorder.h:    EFFICIENCY_CULL    = 0x1 << 14, 
    ./examples/Geant4/OpticalApp/OpticalRecorder.h:    EFFICIENCY_COLLECT = 0x1 << 15, 
    ./examples/Geant4/OpticalApp/OpticalRecorder.h:    static constexpr const char* _EFFICIENCY_COLLECT = "EC" ;
    ./examples/Geant4/OpticalApp/OpticalRecorder.h:    static constexpr const char* _EFFICIENCY_CULL    = "EX" ;
    ./examples/Geant4/OpticalApp/OpticalRecorder.h:        case EFFICIENCY_CULL:    str=_EFFICIENCY_CULL ;break;
    ./examples/Geant4/OpticalApp/OpticalRecorder.h:        case EFFICIENCY_COLLECT: str=_EFFICIENCY_COLLECT ;break;
    ./examples/Geant4/BoundaryStandalone/G4OpBoundaryProcessTest.cc:    eload(qvals3(reflectivity,efficiency,transmittance, "REFLECTIVITY_EFFICIENCY_TRANSMITTANCE", "1,0,0")),
    ./examples/Geant4/BoundaryStandalone/G4OpBoundaryProcessTest.sh:export REFLECTIVITY_EFFICIENCY_TRANSMITTANCE=$ret
    ./optixrap/cu/generate.cu:        p.flags.u.w |= ( u_angular < efficiency ?  EFFICIENCY_COLLECT : EFFICIENCY_CULL ) ;   




Old notes
---------

./optixrap/cu/generate.cu::

    864 #ifdef WITH_SENSORLIB
    865     if( s.flag == SURFACE_DETECT )
    866     {
    867         const unsigned& sensorIndex = s.identity.w ;   // should always be > 0 as flag is SD
    868 #ifdef ANGULAR_ENABLED
    869         const float& f_theta = prd.f_theta ;
    870         const float& f_phi = prd.f_phi ;
    871         const float efficiency = OSensorLib_combined_efficiency(sensorIndex, f_phi, f_theta);
    872         //rtPrintf("//SD sensorIndex %d f_theta %f f_phi %f efficiency %f \n", sensorIndex, f_theta, f_phi, efficiency );
    873 #else
    874         const float efficiency = OSensorLib_simple_efficiency(sensorIndex);
    875         //rtPrintf("//SD sensorIndex %d efficiency %f \n", sensorIndex, efficiency );
    876 #endif
    877         float u_angular = curand_uniform(&rng) ;
    878         p.flags.u.w |= ( u_angular < efficiency ?  EFFICIENCY_COLLECT : EFFICIENCY_CULL ) ;
    879     }
    880 #endif



./g4ok/G4Opticks.cc::

     814 /**
     815 G4Opticks::setSensorData
     816 ---------------------------
     817 
     818 Calls to this for all sensor_placements G4PVPlacement provided by G4Opticks::getSensorPlacements
     819 provides a way to associate the Opticks contiguous 1-based sensorIndex with a detector 
     820 defined sensor identifier. 
     821 
     822 Within JUNO simulation framework this is used from LSExpDetectorConstruction::SetupOpticks.
     823 
     824 sensorIndex 
     825     1-based contiguous index used to access the sensor data, 
     826     the (index-1) must be less than the number of sensors
     827 efficiency_1 
     828 efficiency_2
     829     two efficiencies which are multiplied together with the local angle dependent efficiency 
     830     to yield the detection efficiency used together with a uniform random to set the 
     831     EFFICIENCY_COLLECT (EC) or EFFICIENCY_CULL (EX) flag for photons that already 
     832     have SURFACE_DETECT flag 
     833 category
     834     used to distinguish between sensors with different theta-phi textures   
     835 identifier
     836     detector specific integer representing a sensor, does not need to be contiguous
     837 
     838 
     839 Why call G4Opticks::setSensorData ?
     840 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     841 
     842 Not everything is in GDML.  Detector simulation frameworks often add things on top 
     843 for example local theta and/or phi dependent sensor efficiencies and additional 
     844 efficiency factors.  Also detectors often use there own numbering schemes for sensors. 
     845 That is what the sensor_identifier is. 
     846 
     847 Normally after hits are collected detector simulation frameworks cull them 
     848 randomly based on efficiencies. G4Opticks::setSensorData allows that culling 
     849 to effectively be done on the GPU so the CPU side memory requirements can be reduced 
     850 by a factor of the  efficiency. Often that is something like a quarter of the memory 
     851 reqirements. It also correspondingly reduces the volume of hit data that needs to be copied 
     852 from GPU to CPU.
     853 
     854 **/
     855 
     856 void G4Opticks::setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int category, int identifier)
     857 {
     858     assert( m_sensorlib );
     859     m_sensorlib->setSensorData(sensorIndex, efficiency_1, efficiency_2, category, identifier);
     860 }

 

runtime_angular_control
=========================

see also
----------

* :doc:`runtime_way_control`


WITH_ANGULAR
----------------


::

    epsilon:opticks blyth$ opticks-f WITH_ANGULAR
    ./optickscore/OpticksSwitches.h:#define WITH_ANGULAR 1
    ./optickscore/OpticksSwitches.h:#ifdef WITH_ANGULAR
    ./optickscore/OpticksSwitches.h:    ss << "WITH_ANGULAR " ;   
    ./optickscore/OpticksCfg.cc:       ("angular",  "enable GPU side angular efficiency culling, requires the WITH_ANGULAR compile time switch to be enabled") ;

    ./optixrap/CMakeLists.txt:set(flags_AW +WITH_ANGULAR,+WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_Aw +WITH_ANGULAR,-WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_aW -WITH_ANGULAR,+WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_aw -WITH_ANGULAR,-WAY_ENABLED)

    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR

     wheeling in OSensorLib_ buffers and functions
     hmm that is not angular info specific ?  


    ./optixrap/cu/preprocessor.py:of flag macros, eg -WITH_ANGULAR,+WAY_ENABLED 
    ./optixrap/cu/preprocessor.py:    parser.add_argument( "-f", "--flags", help="Comma delimited control flags eg +WAY_ENABLED,-WITH_ANGULAR " )

    ./optixrap/cu/preprocessor.sh:+WITH_ANGULAR,+WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:+WITH_ANGULAR,-WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:-WITH_ANGULAR,+WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:-WITH_ANGULAR,-WAY_ENABLED

    ./optixrap/OContext.cc:        << a << "WITH_ANGULAR"
    epsilon:opticks blyth$ 


::

     12 OSensorLib::OSensorLib(const OCtx* octx, const SensorLib* sensorlib)
     13     :
     14     m_octx(octx),
     15     m_sensorlib(sensorlib),
     16     m_sensor_data(m_sensorlib->getSensorDataArray()),
     17     m_angular_efficiency(m_sensorlib->getSensorAngularEfficiencyArray()),
     18     m_num_dim(   m_angular_efficiency ? m_angular_efficiency->getNumDimensions() : 0),
     19     m_num_cat(   m_angular_efficiency ? m_angular_efficiency->getShape(0) : 0),
     20     m_num_theta( m_angular_efficiency ? m_angular_efficiency->getShape(1) : 0),
     21     m_num_phi(   m_angular_efficiency ? m_angular_efficiency->getShape(2) : 0),
     22     m_num_elem(  m_angular_efficiency ? m_angular_efficiency->getShape(3) : 0),
     23     m_texid( NPY<int>::make(m_num_cat, 4) )    // small buffer of texid, NB empty when no angular efficiency     
     24 {
     25     init();
     26 }






optickscore/SensorLib
    holder of efficiency table for each sensor
    and possibly angular efficiencies if SensorLib::setSensorAngularEfficiency is called

closest_hit_propagate.cu/closest_hit_angular_propagate.cu
    PRD change adding f_theta f_phi when have angular info 



WITH_ANGULAR
    angular efficiency culling  --angular Opticks::isAngularEnabled


* Huh surprised by how few WITH_ANGULAR
* looks like conflation between having the SensorLib efficiencies and having the extra angular textures

* WITH_ANGULAR/ANGULAR_ENABLED is needed 

* many uses of WITH_ANGULAR should be WITH_SENSORLIB 

* Looks like the "--angular" option and Opticks::isAngularEnabled should be removed
  as having angular efficiency is dictacted by whether SensorLib::setSensorAngularEfficiency gets 
  called via G4Opticks::setSensorAngularEfficiency 

* could have some other angular option to control application of culling, but not to 
  control the generate.cu preprocessor.py flags  


* What switching is actually needed ? 

  * angular eff means need f_theta, f_phi in PRD : this is expensive so should only be there when needed 
  * some WITH_ANGULAR should be WITH_SENSORLIB and compile time is fine, as expect it will almost always be true ? 



Switching between cu/closest_hit_propagate.cu cu/closest_hit_angular_propagate.cu::

     569 optix::Material OGeo::makeMaterial()
     570 {
     571     LOG(verbose)
     572         << " radiance_ray " << OContext::e_radiance_ray
     573         << " propagate_ray " << OContext::e_propagate_ray
     574         ;
     575 
     576     LOG(LEVEL) << "[" ;
     577     optix::Material material = m_context->createMaterial();
     578     material->setClosestHitProgram(OContext::e_radiance_ray, m_ocontext->createProgram("material1_radiance.cu", "closest_hit_radiance"));
     579 
     580     bool angular = m_ok->isAngularEnabled() ;
     581     const char* ch_module = angular ? "closest_hit_angular_propagate.cu" : "closest_hit_propagate.cu" ;
     582     const char* ch_func   = angular ? "closest_hit_angular_propagate"    : "closest_hit_propagate" ;
     583 
     584     LOG(LEVEL)
     585         << " angular " << angular
     586         << " ch_module " << ch_module
     587         << " ch_func " << ch_func
     588         ;
     589     material->setClosestHitProgram(OContext::e_propagate_ray, m_ocontext->createProgram(ch_module, ch_func));
     590 
     591     LOG(LEVEL) << "]" ;
     592     return material ;
     593 }


generate.cu::

    806 #ifdef WITH_SENSORLIB
    807     if( s.flag == SURFACE_DETECT )
    808     {
    809         const unsigned& sensorIndex = s.identity.w ;   // should always be > 0 as flag is SD
    810 #ifdef WITH_ANGULAR
    811         const float& f_theta = prd.f_theta ;
    812         const float& f_phi = prd.f_phi ; 
    813         const float efficiency = OSensorLib_combined_efficiency(sensorIndex, f_phi, f_theta);
    814         //rtPrintf("//SD sensorIndex %d f_theta %f f_phi %f efficiency %f \n", sensorIndex, f_theta, f_phi, efficiency );
    815 #else   
    816         const float efficiency = OSensorLib_simple_efficiency(sensorIndex);
    817         //rtPrintf("//SD sensorIndex %d efficiency %f \n", sensorIndex, efficiency );
    818 #endif  
    819         float u_angular = curand_uniform(&rng) ;
    820         p.flags.u.w |= ( u_angular < efficiency ?  EFFICIENCY_COLLECT : EFFICIENCY_CULL ) ;
    821     }   
    822 #endif
    823 
      


WITH_ANGULAR -> ANGULAR_ENABLED
----------------------------------

* replace the compile time switch with an oxrap/cu/preprocessor.py ANGULAR_ENABLED flag 

::

    epsilon:optickscore blyth$ opticks-f WITH_ANGULAR
    ./optickscore/OpticksSwitches.h:#define WITH_ANGULAR 1
    ./optickscore/OpticksSwitches.h:#ifdef WITH_ANGULAR
    ./optickscore/OpticksSwitches.h:    ss << "WITH_ANGULAR " ;   
    ./optixrap/CMakeLists.txt:set(flags_AW +WITH_ANGULAR,+WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_Aw +WITH_ANGULAR,-WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_aW -WITH_ANGULAR,+WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_aw -WITH_ANGULAR,-WAY_ENABLED)
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/generate.cu:#ifdef WITH_ANGULAR
    ./optixrap/cu/preprocessor.py:of flag macros, eg -WITH_ANGULAR,+WAY_ENABLED 
    ./optixrap/cu/preprocessor.py:    parser.add_argument( "-f", "--flags", help="Comma delimited control flags eg +WAY_ENABLED,-WITH_ANGULAR " )
    ./optixrap/cu/preprocessor.sh:+WITH_ANGULAR,+WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:+WITH_ANGULAR,-WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:-WITH_ANGULAR,+WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:-WITH_ANGULAR,-WAY_ENABLED
    ./optixrap/OContext.cc:        << a << "WITH_ANGULAR"
    epsilon:opticks blyth$ 


    epsilon:opticks blyth$ opticks-f ANGULAR_ENABLED
    ./optixrap/CMakeLists.txt:set(flags_AW +ANGULAR_ENABLED,+WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_Aw +ANGULAR_ENABLED,-WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_aW -ANGULAR_ENABLED,+WAY_ENABLED)
    ./optixrap/CMakeLists.txt:set(flags_aw -ANGULAR_ENABLED,-WAY_ENABLED)
    ./optixrap/cu/generate.cu:#ifdef ANGULAR_ENABLED
    ./optixrap/cu/generate.cu:#ifdef ANGULAR_ENABLED
    ./optixrap/cu/generate.cu:#ifdef ANGULAR_ENABLED
    ./optixrap/cu/generate.cu:#ifdef ANGULAR_ENABLED
    ./optixrap/cu/preprocessor.py:of flag macros, eg -ANGULAR_ENABLED,+WAY_ENABLED 
    ./optixrap/cu/preprocessor.py:    parser.add_argument( "-f", "--flags", help="Comma delimited control flags eg +WAY_ENABLED,-ANGULAR_ENABLED " )
    ./optixrap/cu/preprocessor.sh:+ANGULAR_ENABLED,+WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:+ANGULAR_ENABLED,-WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:-ANGULAR_ENABLED,+WAY_ENABLED
    ./optixrap/cu/preprocessor.sh:-ANGULAR_ENABLED,-WAY_ENABLED
    ./optixrap/OContext.cc:        << a << "ANGULAR_ENABLED"
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 

 

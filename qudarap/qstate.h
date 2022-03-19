#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSTATE_METHOD __device__
#else
   #define QSTATE_METHOD 
#endif 

/**
qstate.h
=========

Populated by qsim::fill_state from texture and buffer lookups 
using photon wavelength and the boundary obtained from geometry 
intersect.  

Old version of this in OptiXRap/cu also copied things from "PRD" into here ... 
BUT seems no point doing that, can just directly use them from PRD. 

**/

struct qstate
{
    float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
    float4 m1group2 ;     // group_velocity/spare1/spare2/spare3
    float4 material2 ;   
    float4 surface ;      // detect/absorb/reflect_specular/reflect_diffuse

    uint4  optical ;      // x/y/z/w index/type/finish/value  
    uint4  index ;        // indices of m1/m2/surf/sensor

#ifdef WAY_ENABLED
    float4 way0 ;   
    float4 way1 ;   
#endif

    // unsigned flag ; 
    // float3 surface_normal ; 
    // float distance_to_boundary ;
    // uint4 identity ;  //  node/mesh/boundary/sensor indices of last intersection
    // unsigned identity ;  
};




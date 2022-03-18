#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QSTATE_METHOD __device__
#else
   #define QSTATE_METHOD 
#endif 


struct qstate
{
   // primary group populated by qsim::fill_state with texture and buffer lookups 
   // based in on the wavelength and boundary obtained from intersect 

   float4 material1 ;    // refractive_index/absorption_length/scattering_length/reemission_prob
   float4 m1group2  ;    // group_velocity/spare1/spare2/spare3
   float4 material2 ;   
   float4 surface    ;   //  detect/absorb/reflect_specular/reflect_diffuse
   uint4 optical ;   // x/y/z/w index/type/finish/value  
   uint4 index ;     // indices of m1/m2/surf/sensor

   // this lots are just convenient transients 
   //unsigned flag ; 
   //float3 surface_normal ; 
   //float distance_to_boundary ;
   //uint4 identity ;  //  node/mesh/boundary/sensor indices of last intersection
   //unsigned identity ;  

#ifdef WAY_ENABLED
   float4 way0 ;   
   float4 way1 ;   
#endif


};






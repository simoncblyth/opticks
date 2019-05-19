equirectangular_camera_control
===============================

Context :doc:`equirectangular_camera_blackholes_sensitive_to_far`


issue
------

::

   geocache-;geocache-360

   B,B : make projective PMTs visible
   O   : switch to raytrace
   D,D : ORTHOGRAPHIC then EQUIRECTANGULAR  



Its not a full 360 view, as can only see one pole (maybe just 180).

::

     54 RT_PROGRAM void pinhole_camera()
     55 {
     56 
     57   PerRayData_radiance prd;
     58   prd.flag = 0u ; 
     59   prd.result = bad_color ;
     60 
     61   float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;   // (-1:1, -1:1 ) 
     62 
     63 
     64   optix::Ray ray ;
     65 
     66   if( parallel == 0u ) // PERSPECTIVE_CAMERA
     67   {
     68       float3 ray_origin    = eye                          ;
     69       float3 ray_direction = normalize(d.x*U + d.y*V + W) ;
     70       ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
     71   }
     72   else if ( parallel == 1u )  // ORTHOGRAPHIC_CAMERA
     73   {
     74       float3 ray_origin    = eye + d.x*U + d.y*V ;
     75       float3 ray_direction = normalize(W)        ;
     76       ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
     77   }
     78   else if ( parallel == 2u ) // EQUIRECT_CAMERA
     79   {
     80       // OptiX/SDK/optixTutorial/tutorial11.cu:env_camera
     81       // https://www.shadertoy.com/view/XsBSDR
     82       //
     83       //
     84       // azimuthal angle "phi"   :  -pi   -> pi
     85       // polar angle     "theta" :  -pi/2 -> pi/2
     86
    


 
     87       float2 azipol = make_float2(launch_index) / make_float2(launch_dim) * make_float2(2.0f*M_PIf , M_PIf ) ; // + make_float2( M_PIf, M_PIf/2.0f ) ; 
     88       float3 angle = make_float3(cos(azipol.x) * sin(azipol.y), -cos(azipol.y), sin(azipol.x) * sin(azipol.y));
     89       //                     cos(azi) sin(pol) , -cos(pol),   sin(azi)cos(pol) 
     90 
     91       //float3 angle = make_float3( sin(azipol.y) * cos(azipol.x), sin(azipol.y) * sin(azipol.x),  cos(azipol.y) ) ;
     92       // conventional spherical to cartesian 
     93 
     94 
     95       float3 ray_origin    = eye ;
     96       float3 ray_direction = normalize(angle.x*normalize(U) + angle.y*normalize(V) + angle.z*normalize(W));
     97 
     98       ray = optix::make_Ray( ray_origin , ray_direction, radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX) ;
     99   }







benchmarks360
=================

* https://stackoverflow.com/questions/44082298/how-to-make-360-video-output-in-opengl

A 360-surround view from the center of the scintillator would make a 
good benchmark : when switch off the globals anyhow.

Although easy in a ray tracer doing it in OpenGL starting from vertices looks
to be far too difficult for the purpose of a benchmark : is equivalent to 
making 360 degree video by stitching together different views into texture.

See env- equirect- for some reading around about this


Ray tracing part is easy
---------------------------

::

    [blyth@localhost SDK-precompiled-samples]$ LD_LIBRARY_PATH=. ./optixTutorial -T 11
    [blyth@localhost SDK-precompiled-samples]$ pwd
    /home/blyth/local/opticks/externals/OptiX/SDK-precompiled-samples

::

     46 //
     47 // (NEW)
     48 // Environment map camera
     49 //
     50 rtDeclareVariable(float3,        eye, , );
     51 rtDeclareVariable(float3,        U, , );
     52 rtDeclareVariable(float3,        V, , );
     53 rtDeclareVariable(float3,        W, , );
     54 rtDeclareVariable(float3,        bad_color, , );
     55 rtBuffer<uchar4, 2>              output_buffer;
     56 RT_PROGRAM void env_camera()
     57 {
     58   size_t2 screen = output_buffer.size();
     59 
     60   float2 d = make_float2(launch_index) / make_float2(screen) * make_float2(2.0f * M_PIf , M_PIf) + make_float2(M_PIf, 0);
     //    map pixels onto phi,theta 

     61   float3 angle = make_float3(cos(d.x) * sin(d.y), -cos(d.y), sin(d.x) * sin(d.y));
     //                              cos(phi)*sin(theta), -cos(theta), sin(phi)*sin(theta)  
     

     62   float3 ray_origin = eye;
     63   float3 ray_direction = normalize(angle.x*normalize(U) + angle.y*normalize(V) + angle.z*normalize(W));
     64 
     65   optix::Ray ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon);
     66 
     67   PerRayData_radiance prd;
     68   prd.importance = 1.f;
     69   prd.depth = 0;
     70 
     71   rtTrace(top_object, ray, prd);
     72 
     73   output_buffer[launch_index] = make_color( prd.result );
     74 }
     75 


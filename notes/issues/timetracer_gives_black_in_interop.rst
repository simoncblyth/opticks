timetracer_gives_black_in_interop
===================================


optixrap/cu/pinhole_camera.cu::

    RT_PROGRAM void pinhole_camera_timetracer()
    {

      PerRayData_radiance prd;
      prd.flag = 0u ; 
      prd.result = bad_color ;

      float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;

      optix::Ray ray = parallel == 0 ? 
                           optix::make_Ray( eye                 , normalize(d.x*U + d.y*V + W), radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
                         :
                           optix::make_Ray( eye + d.x*U + d.y*V , normalize(W)                , radiance_ray_type, scene_epsilon, RT_DEFAULT_MAX)
                         ;

      clock_t t0 = clock(); 

      rtTrace(top_object, ray, prd);

      clock_t t1 = clock(); 

      float dt = ( t1 - t0 ) ; 

      float pixel_time  = dt * timetracerscale ;

      uchar4  color = make_color( make_float3( pixel_time )); 

      rtPrintf("//pinhole_camera_timetracer dt %10.3f pixel_time %10.3f timetracerscale %10.3g color (%d %d %d) \n", dt, pixel_time, timetracerscale, color.x, color.y, color.z );  

      //uchar4  color = RED ;  

      output_buffer[launch_index] = color ; 
    }



Looks like reasonable color values, but just gives a black screen in interop::

    [blyth@localhost issues]$ cvd 1 OKTest --rtx 0 --timetracer --timetracerscale 5e-6 --pindex 0

    /pinhole_camera_timetracer dt 159767.000 pixel_time      0.799 timetracerscale      5e-06 color (204 204 204) 
    //pinhole_camera_timetracer dt 141843.000 pixel_time      0.709 timetracerscale      5e-06 color (181 181 181) 
    //pinhole_camera_timetracer dt 106079.000 pixel_time      0.530 timetracerscale      5e-06 color (135 135 135) 
    //pinhole_camera_timetracer dt 110390.000 pixel_time      0.552 timetracerscale      5e-06 color (141 141 141) 
    //pinhole_camera_timetracer dt 103854.000 pixel_time      0.519 timetracerscale      5e-06 color (132 132 132) 
    //pinhole_camera_timetracer dt  93793.000 pixel_time      0.469 timetracerscale      5e-06 color (120 120 120) 
    //pinhole_camera_timetracer dt 111078.000 pixel_time      0.555 timetracerscale      5e-06 color (142 142 142) 
    //pinhole_camera_timetracer dt 104592.000 pixel_time      0.523 timetracerscale      5e-06 color (133 133 133) 
    //pinhole_camera_timetracer dt 108361.000 pixel_time      0.542 timetracerscale      5e-06 color (138 138 138) 
    //pinhole_camera_timetracer dt 107988.000 pixel_time      0.540 timetracerscale      5e-06 color (138 138 138) 
    2019-04-26 22:54:59.535 INFO  [435337] [Frame::key_pressed@823] Frame::key_pressed escape
    2019-04-26 22:54:59.536 INFO  [435337] [Opticks::dumpRC@196]  rc 0 rcmsg : -


Ordinary raytrace works::

    cvd 1 OKTest --rtx 0 --xanalytic --gltf 1

Which is bizarre as its almost exactly the same machinery.


As the display is connected to TITAN RTX need to use "cvd 1" when using interop.::

    cvd () 
    { 
        local devs=$1;
        shift;
        CUDA_VISIBLE_DEVICES=$devs $*
    }




But it works fine for compute snaps::

    geocache-check --timetracer --pindex 0
    geocache-check --timetracer --pindex 0 --timetracerscale 5e-7

::

    geocache-check is a function
    geocache-check () 
    { 
        local stamp=$(date +%s);
        local rtx=0;
        CUDA_VISIBLE_DEVICES=0 geocache-bench- --rtx $rtx --runfolder $FUNCNAME --runstamp $stamp --runlabel "R${rtx}_TITAN_V" $*
    }


    geocache-bench- is a function
    geocache-bench- () 
    { 
        type $FUNCNAME;
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=5,eyestartz=-1,eyestopz=-0.5" --size 5120,2880,1 --embedded $*
    }



FIXED : interop-flipping-to-raytrace-gives-GL-error-invalid-operation-at-rtContextLaunch2D 
================================================================================================

issue
------

geocache-load succeeds to load and visualize the propagation,

BUT

1. starting viewpoint is very close to the center
2. **switching on raytrace render crashes with : Assertion failed: "GL error: Invalid operation"**

::


    [blyth@localhost optickscore]$ t geocache-load
    geocache-load is a function
    geocache-load () 
    { 
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OKTest --cvd 0 --envkey --xanalytic --timemax 400 --animtimemax 400 --load
    }


    2019-05-10 17:47:03.152 INFO  [183662] [Scene::setRecordStyle@1125] vector
    2019-05-10 17:47:04.973 INFO  [183662] [GlobalStyle::setGlobalStyle@97] GlobalStyle GVIS
    2019-05-10 17:47:28.508 FATAL [183662] [ContentStyle::setContentStyle@98] ContentStyle bbox inst 0 bbox 1 wire 0 asis 0 m_num_content_style 0 NUM_CONTENT_STYLE 5
    2019-05-10 17:47:29.125 FATAL [183662] [ContentStyle::setContentStyle@98] ContentStyle norm inst 1 bbox 0 wire 0 asis 0 m_num_content_style 0 NUM_CONTENT_STYLE 5
    2019-05-10 17:48:35.769 INFO  [183662] [RenderStyle::setRenderStyle@95] RenderStyle R_COMPOSITE
    2019-05-10 17:48:36.114 INFO  [183662] [OTracer::trace_@140]  entry_index 0 trace_count 0 resolution_scale 1 size(1920,1080) ZProj.zw (-1.00401,-283.41) front 0.4198,0.6826,-0.5981
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Assertion failed: "GL error: Invalid operation
    ")
    Aborted (core dumped)


Probable cause : was led astray by nvidia-smi slot not matching the OptiX one 
-----------------------------------------------------------------------------------

See :doc:`nvidia-smi-slot-index-doesnt-match-UseOptiX-index-and-may-change-on-reboot`

::

    [blyth@localhost issues]$ UseOptiX --cvd 0
    setting envvar internally : CUDA_VISIBLE_DEVICES=0
    OptiX version 60000 major.minor.micro 6.0.0   Number of devices = 1 

     Device 0:                        TITAN V  Compute Support: 7 0  Total Memory: 12621381632 bytes 


When using interop have to check are making the GPU connected to the display the only visible one, 
according to UseOptiX (not nvidia-smi)::

    geocache-load () 
    { 
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        local cvd=1;
        UseOptiX --cvd $cvd;
        $dbg OKTest --cvd $cvd --rtx 0 --envkey --xanalytic --timemax 400 --animtimemax 400 --load
    }



BUT OKTest flip to raytrace works
-------------------------------------

::

    OKTest     ## defaults to old route DYB triangulates
    # "a" to start propagation
    # "o" to flip on the raytrace  


    OKTest --envkey       ## JUNO triangulated flip to raytrace works

    OKTest --envkey --xanalytic   ## huh analytic worked too

    OKTest --envkey --xanalytic --load   ## now working too with a loaded propagation
  
   


nvidia-smi GPU slot index doesnt match the UseOptiX one 
------------------------------------------------------------ 



workaround for making snaps is OpSnapTest pure compute snaps
-------------------------------------------------------------







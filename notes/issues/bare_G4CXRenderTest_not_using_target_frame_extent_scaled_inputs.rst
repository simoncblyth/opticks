bare_G4CXRenderTest_not_using_target_frame_extent_scaled_inputs.rst
=====================================================================


The bare G4CXRenderTest executable without any script environment control
is not using extent scaled target inputs like EYE. 
So probably need to change some defaults regarding target frame, 
so can use bare exeutable more easily. 
Have to specify EYE absolute and TMIN relative it appears
to get reasonable full geom renders::

   EYE=-40000,0,0 TMIN=0.5 G4CXRenderTest

::

   scp P:/data/simon/opticks/GEOM/V1J011/G4CXRenderTest/nonamepfx_0.jpg .
   open nonamepfx_0.jpg 

::

    [simon@localhost tests]$ EYE=-40000,0,0 TMIN=0.5 G4CXRenderTest
    2023-11-09 16:56:54.942 INFO  [344795] [main@19] [ cu first 
    2023-11-09 16:56:55.194 INFO  [344795] [main@21] ] cu first 
    2023-11-09 16:56:55.977 INFO  [344795] [CSGOptiX::render@1070] /data/simon/opticks/GEOM/V1J011/G4CXRenderTest/nonamepfx_0.jpg :     0.0110 0:NVIDIA_TITAN_V 1:NVIDIA_TITAN_RTX 






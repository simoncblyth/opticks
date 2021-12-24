making_flightpath_raytrace_movies
===================================

Overview
----------

Opticks includes several scripts and executables that can 
be used to make movies of geometry and geometry+event propagations 
using various techniques based on both raytraced and rasterized visualizations.

Some techniques are based on the combination of .jpg images into .mp4 using ffmpeg 
whilst some use separate screen capture software to capture movies from Opticks OpenGL windows.  

* NVIDIA OptiX pre-7 ray traced geometry 
* OpenGL rasterized event propagations (implemented somewhat confusingly with OpenGL geometry shaders) composited with NVIDIA OptiX pre-7 geometry   
* OpenGL rasterized event propagations combined with OpenGL rasterized representations of Geant4 polygonized geometry 
* NVIDIA OptiX 7 ray traced geometry 

This page seeks to catalog and organized the various approaches and briefly document them. 

Related
---------

* :doc:`python_browsing_geometry`


Scripts
----------

bin/flight.sh 
   uses executable OpFlightPathTest okop/tests/OpFlightPathTest.cc
   jpg created by the executable are combined into mp4 using ffmpeg
    
bin/flight7.sh 
   previously used CSGOptiXFlight, probably now CSGOptiXRenderTest CSGOptiX/tests/CSGOptiXRenderTest.cc
   jpg created by the executable are combined into mp4 using ffmpeg

   TODO: CFBASE location changes need attention  







Old Notes
-----------

2. create an eye-look-up flight path, that is saved to /tmp/flightpath.npy::

   flight.sh --roundaboutxy 

3. launch visualization, press U to switch to the animated InterpolatedView created from the flightpath::

   OTracerTest --targetpvn lFasteners_phys
   OTracerTest --target 69078

4. for non-interative raytrace jpg snaps around the flightpath::

   PERIOD=8 PVN=lLowerChimney_phys EMM=~5, flight.sh --rtx 1 --cvd 1 

5. make an mp4 from the jpg snaps.  flight.sh can automatically 
   create the mp4 assuming env repo and ffmpeg are installed

6. when doing the above snaps on remote ssh node P::

   okop ; cd tests
   ./OpFlightPathTest.sh grab 


record_animation_render_debugging_tips
========================================



Settings for debug
-------------------

FULLSCREEN=0 cxr_min.sh
    disabling FULLSCREEN allows to see simulation time in window title



Interactive controls to try to make record renders visible
-------------------------------------------------------------


alt-O
   toggle rendering of geometry

   * very useful to make record points more visible - AS OFTEN OBSCURED


alt-A
   enable photon record aimimation rendering of AFOLD/record.npy

alt-B
   enable photon record aimimation rendering of BFOLD/record.npy


alt-T
   reset time back to T0

ctrl-T
   toggle animation time progression - ie stop/start time


O
   toggle orthographic/perspective projection

W/S
   forward backwards viewpoint control, does nothing in orthographic mode

Z
   toggle zoom control - then drag mouse up/down to adjust,
   zoon often needed in orthographic mode to make all geom visible



see also
----------

* ~/o/g4cx/tests/G4CXTest_raindrop_animation.sh
* ~/o/examples/UseGeometryShader/go.sh





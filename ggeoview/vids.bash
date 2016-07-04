# === func-gen- : graphics/ggeoview/vids fgp graphics/ggeoview/vids.bash fgn vids fgh graphics/ggeoview
vids-src(){      echo graphics/ggeoview/vids.bash ; }
vids-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(vids-src)} ; }
vids-vi(){       vi $(vids-source) ; }
vids-env(){      olocal- ; }
vids-usage(){ cat << EOU

GGeoView Video Captures 
=============================

* See Also :doc:`graphics/ggeoview/movie`

Capture Procedure
--------------------

1. Exit all apps, leaving just Terminal.app and Finder.app 

2. Launch the visualization with either::

   ggv-;ggv-dyb
   ggv-;jpmt

3. **DO NOT MOVE** the ggv window, avoid terminal windows overlapping it  

4. Invoke Quicktime capture with:: 

   ggv-hd-capture 

   * does not always work, possibly starting Quicktime Player first
     avoids this issue

   * the Quicktime Screen Capture little window pops up, 
     a few seconds later the screen should dim leaving
     just the ggv window brightened and surrounded by a dotted line, 
     with a central button to begin recording.

   * tap GGV window to adjust focus back to GGV, so key shortcuts work

   * stop the capture with ctrl-command-escape

   * save the .mov by entering a path when closing the Quicktime 
     window, otherwise discards the file

Video Format
-------------

ggv-hd-capture succeeds to create::

   Format H.264   1920x1080
   FPS            60
   Data rate      25.92 Mbit/s


Observations 
-----------------

* it takes a while for recording to start, 
  so start well before the desired entry and later
  use iMovie to set the precise clip start/end points

Issues
--------

* Selecting "do not show cursor" seems not to be obeyed
* playback appears choppy, maybe some combination of animation period, animation speed 
  and iMovie speed up can mitigate ? Or might just need a desktop GPU to do this smoothly.


Vids
-----

::

    simon:Movies blyth$ du -h *.mov

    209M    dyb_geometry.mov
     63M    dyb_geometry_event_01.mov
    101M    dyb_track_01.mov
    147M    jpmt_iv_02.mov
     67M    jpmt_tv_01.mov


1. dyb_geometry.mov

   B: PMTs
   E: normal shader 
   U1: InterpolatedView fly around water pool
   O: optix raytrace view
   O2: back to OpenGL 
   U1 continued: fly thru AD 
   C: cut away view

2. dyb_geometry_event_01.mov

   B: PMTs
   E: normal shader 
   4: bookmark AD view
   C: cutaway
   G2: adjust Camera near to about 80mm
   G1: GUI off
   M1: mat1

3. dyb_track_01.mov

   B: PMTs
   E: normal shader 
   U3 : track view

4. jpmt_iv_02.mov

   Version 02 uses full PMT all the way for constant speedup
   of entire clip in iMovie

   E: normal shader
   Q: invis global
   L: flip normals
   B: switch to PMTs, as approach 

   U: InterpolatedView 
   T: faster
   L: flip normals, once inside
   A: start event


5. jpmt_tv_01.mov

   E,Q
   U3: TrackView 
   B
   r: adjust direction down to left
   G/View: fraction scale:1.7 get ahead track gfront



Thoughts for more vids

* orbital view
* orthographic view


YouTube Video Sharing
----------------------

Shared from iMovie, after sign on to YouTube with google account and selected 

* initial attemt failed, it is necessary to sign on to YouTube from web first 
  (with google account) and create a channel, subsequent share from iMovie worked

* initially the video appears in low resolution only, after 30min or so 
  a HD version becomes visible

* it is necessary to adjust to being Public in web interface 


Opticks_GTC_001

* https://www.youtube.com/watch?v=QzH6y0pKXk4 


Chinese Accessible Video ?
---------------------------

Chinese eqivalent tudou works with chrome translation

* http://www.tudou.com
* http://www.youku.com


YouKou 
---------

* http://yktips.com/how-to-create-youku-account-and-register/

* seems youku requires a chinese phone number to register



EOU
}
vids-dir(){ echo $(local-base)/env/graphics/ggeoview/graphics/ggeoview-vids ; }
vids-cd(){  cd $(vids-dir); }
vids-mate(){ mate $(vids-dir) ; }
vids-get(){
   local dir=$(dirname $(vids-dir)) &&  mkdir -p $dir && cd $dir

}

Movie
======

Thinking about recording GLFW events for replay to make movies
-----------------------------------------------------------------

* :doc:`glfw_gleq_event_record_replay`



Experience with g4daeview
----------------------------

* :doc:`geant4/geometry/collada/g4daeview/g4daeview_usage.txt`

With g4daeview created movie using `QuickTime Player.app` 

* `File > New Screen Recording` or command+control+N

   * dialog pops up : click to record entire screen, drag to record part of screen
   * creates a very large .mov (~1GB for ~2min) 
   * stop recording using : command+control+escape 

* `File > Export ...` to compress .mov to .m4v 


Resulting movies like the below however need editing to avoid boring waits between action.

* /Users/blyth/Dropbox/Public/g4daeview_001.m4v


References
-----------

* :google:`iMovie Quicktime App`

  * https://developer.apple.com/app-store/app-previews/imovie/Creating-App-Previews-with-iMovie.pdf

* :google:`iMovie import screen capture`

* http://www.makeuseof.com/tag/5-best-screen-recorders-capturing-mac-os-x/


op.sh event movies
-------------------

::

    op.sh --j1707 --gltf 3 --cerenkov --compute --save
         ## using cerenkov genstep computes and saves the OpticksEvent (photon propagation)

    op.sh --j1707 --gltf 3 --cerenkov --load
         ## loads and visualizes the persisted propagation

    op.sh --j1707 --gltf 3 --scintillation --compute --timemax 400 --animtimemax 400




* see bin/ggv.sh for some examples, need to set timemax to avoid running out of time mid CD


QuickTime Player 10.3
-----------------------

* curiously "View > Actual Size" clips the edges with g4daeview_001.m4v, the movie inspector says:

  * Format: H.264, 1728x1080
  * Current Size: 1440x900 


Retina full is: 2880x1800 dividing by 2: 1440x900

::

    In [2]: 1728./1080.
    Out[2]: 1.6

    In [3]: 16./9.
    Out[3]: 1.7777777777777777


Aspect Ratio
---------------

::

    1:1   1.0 
    5:4   1.25    
    4:3   1.333     Standard 4:3: 320x240, 640x480, 800x600, 1024x768
    8:5   1.6 
    16:9  1.777     Widescreen 16:9: 640x360, 800x450, 960x540, 1024x576, 1280x720, and 1920x1080



What size video ?
---------------------------------------------

* GTC provides projector : 1920x1080p @ 60hz

* what size ggv window to record ?
* when dragging a region, resulting video seems to be twice the pixel size selected, 
  but seems no way to get a precise pixel dimension

* https://forums.creativecow.net/thread/335/37515

* https://www.techsmith.com/tutorial-camtasia-mac-aspect-ratio-current.html


Automated Screen Capture
-------------------------

* http://apple.stackexchange.com/questions/176555/how-do-i-make-multiple-screen-recordings-with-the-exact-same-portion-of-the-scre


QuickTime Applescript
-----------------------

* it seems no way to control the screen area recorded via scripting 
* http://www.neerajkumar.net/blog/2013/02/16/script-to-record-screen-on-mac-osx-and-save-on-disk/


iMovie importing
-----------------

* https://discussions.apple.com/thread/2282019?start=0&tstart=0

  * (circa 2010) implies that .mov (the uncompressed one) is best out of .mov or .m4v  





Movie
======


Experience with g4daeview
----------------------------

* :doc:`geant4/geometry/collada/g4daeview/g4daeview_usage.txt`

With g4daeview created movie using `QuickTime Player.app` 

* `File > New Screen Recording` 

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


Aspect 
--------

::

    1:1   1.0 
    5:4   1.25    
    4:3   1.333   
    8:5   1.6 
    16:9  1.777


What size video ?
---------------------------------------------

* what size ggv window to record ?
* when dragging a region, resulting video seems to be twice the pixel size selected 



QuickTime Applescript
-----------------------

* it seems no way to control the screen area recorded via scripting 
* http://www.neerajkumar.net/blog/2013/02/16/script-to-record-screen-on-mac-osx-and-save-on-disk/


iMovie importing
-----------------

* https://discussions.apple.com/thread/2282019?start=0&tstart=0

  * (circa 2010) implies that .mov (the uncompressed one) is best out of .mov or .m4v  





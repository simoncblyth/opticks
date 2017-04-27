Directory Used for Invoking OKTest Screenshots
================================================


Making Screenshots
-------------------

The .png are copied from Desktop into the the bitbucket static repo  /Users/blyth/simoncblyth.bitbucket.org/env/ok/


::

   osx_
   osx_ss_cp dyb_tboolean_dd_parade


::

    simon:ok blyth$ osx_ss_cp dyb_tboolean_dd_parade
    === osx_ss_cp : iwd /Users/blyth/opticks/ok rel ok repo env dir /Users/blyth/simoncblyth.bitbucket.org/env/ok dst /Users/blyth/simoncblyth.bitbucket.org/env/ok/dyb_tboolean_dd_parade.png
    total 9464
    -rw-r--r--@ 1 blyth  staff  2092223 Jan 15 13:49 dyb_raytrace_composite_cerenkov.png
    -rw-r--r--  1 blyth  staff   908206 Jan 15 13:58 dyb_raytrace_composite_cerenkov_half.png
    -rw-r--r--@ 1 blyth  staff   289854 Jan 15 13:59 dyb_raytrace_composite_cerenkov_half_half.png
    -rw-r--r--@ 1 blyth  staff  1549170 Apr 27 16:38 dyb_tboolean_dd_parade.png
    cp "/Users/blyth/Desktop/Screen Shot 2017-04-27 at 4.34.34 PM.png" /Users/blyth/simoncblyth.bitbucket.org/env/ok/dyb_tboolean_dd_parade.png
    Destination file exists already : enter YES to overwrite YES
    -rw-r--r--@ 1 blyth  staff  1549170 Apr 27 16:39 /Users/blyth/simoncblyth.bitbucket.org/env/ok/dyb_tboolean_dd_parade.png

    .. image:: /env/ok/dyb_tboolean_dd_parade.png
       :width: 900px
       :align: center

    simon:ok blyth$ 



Typically retina screenshots need downsizing for the resolution-challenged world
----------------------------------------------------------------------------------

::

    simon:ok blyth$ downsize.py dyb_tboolean_dd_parade.png
    INFO:env.doc.downsize:Resize 2  
    INFO:env.doc.downsize:downsize dyb_tboolean_dd_parade.png to create dyb_tboolean_dd_parade_half.png 2478px_1616px -> 1239px_808px 
    simon:ok blyth$ 
    simon:ok blyth$ downsize.py dyb_tboolean_dd_parade_half.png
    INFO:env.doc.downsize:Resize 2  
    INFO:env.doc.downsize:downsize dyb_tboolean_dd_parade_half.png to create dyb_tboolean_dd_parade_half_half.png 1239px_808px -> 619px_404px 
    simon:ok blyth$ 



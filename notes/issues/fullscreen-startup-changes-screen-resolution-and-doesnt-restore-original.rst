fullscreen-startup-changes-screen-resolution-and-doesnt-restores-original
==========================================================================

To avoid this issue make sure to use the correct size for your screen resolution::

    geocache-;geocache-360 --fullscreen --size 2560,1440,1 --rtx 1

If forget to do that, logging out and back in will regain your screen resolution.

Note that RTX mode is considerably faster at initialization of geometry, 
but not always faster at raytracing. 


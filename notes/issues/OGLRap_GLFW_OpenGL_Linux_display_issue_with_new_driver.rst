OGLRap_GLFW_OpenGL_Linux_display_issue_with_new_driver
=========================================================

Issue : Linux NVIDIA driver 418.56 glfw-3.2.1 from yum
---------------------------------------------------------------

Launch::

   OKTest 


* Window pops up, pressing Q twice to makes the geometry 
  appear in a mangled form : only along a narrow horizonal band. 

* Pressing R for rotation makes the narrow band of color get bigger
  and fills the screen with triangles emanating from a point on screen : as
  if a very wrong projection is used. 

* Pressing H for home gets back to the narrow band. Subsequently 
  any mouse movement (without any GUI keys enabled) looses the band 
  replaced with a near vertical dotted line. 

* Pressing G for GUI brings it up, and it displays and works normally
  on top of the bizarre geometry render


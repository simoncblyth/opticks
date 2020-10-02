Visualization of Geometry and Event data
==========================================


.. contents:: Table of Contents
   :depth: 2


Visualization Finding Geometry
-------------------------------

Visualization::

   OTracerTest --target 3153 --dbgaim   # just viz, fast start as does no simulation
   OKTest --target 3153 --dbgaim        # does Opticks simulation before viz
   OKG4Test --target 3153 --dbgaim      # does both Geant4 and Opticks simulations before viz

   ## --target volume-index 
   ## --dbgaim : dump some logging about geometry volumes and targetting  


To avoid having to use the target option use an envvar::

   export OPTICKS_DEFAULT_TARGET=3153 
   ## do this together setting the OPTICKS_KEY which picks the geocache


At startup things to try:

* press Q twice, this toggles the global (non-instanced) geometry on/off 
* press B several times, this swiches the render style of the instanced geometry 
* press V, this toggles rotation of the geometry 
* press A, this starts the optical photon propagation animation 
* press O, switches between OpenGL rasterized and OptiX ray trace rendering  

If after all that you still have a black screen, try:

* changing the target to a volume appropriate to your geometry, 
  it defaults to 0 corresponding to the world volume. However 
  maybe you have a very large world volume causing your detector to 
  not be visible.

* press G, this brings up a GUI with help menus etc..



Geometry Tracer Mode
-----------------------

When visualizing geometry that is new to you it is 
best to start with "--tracer" mode as it does not perform 
a simulation, just views geometry.

.. code-block:: sh

    op --dpib --tracer 
    op --dyb  --tracer 
    op --dfar --tracer 
    op --dlin --tracer 
    op --jpmt --tracer 
    op --lxe --tracer 

Sometimes this will result in a black screen, for example
when the viewpoint is within a volume and the "light" is outside 
that volume.  Use the below key controls to make the geometry visible.

::

    V: View::nextMode                  rotate view, with shift modifier rotates in opposite direction 
    L: Composition::nextNormalStyle    flip normal in shaders 
    B: Scene::nextGeometryStyle        bbox/norm/wire 
    C: Clipper::next                   toggle geometry clipping 
    E: Composition::nextGeometryStyle  lightshader/normalshader/flatvertex/facecolor 
    D: Camera::nextStyle               perspective/orthographic 
    O: OptiX render mode               raytrace/hybrid/OpenGL
    G: gui mode                        toggle GUI 


Particularly useful commands for making geometry visible are:

**L**
     flips normals so light from outside a volume can still light inside the volume 
**E** 
     changes the coloring style


For a full list of keys::

    op --keys  


GUI control
-------------

Toggle between GUI modes with the **G** key, the modes are:

* no-GUI
* animation control and time scrubber
* full GUI menu

Sections of the GUI menu:

**Help**
      **Needs Updating**
**Params**
      Information about the loaded event
**Stats**
      **Needs Updating** Can be configured to contain processing timings
**Interactor**
      List of the interactive key shortcuts, the same list 
      can be obtained from commandline with `op.sh --keys` 
**Scene**
      Checkboxes to select what is displayed, **bb** stands for bounding box,
      **in** stands for instance (ie the PMTs)  
**Composition**
      Parameters controlling the view, mostly for developer usage.
**View**
      Sliders to control positions of: **eye**, **look** and **up**
**Camera**
      Sliders controlling **near**, **far**, **zoom** and **scale** of the 
      camera and a toggle inbetween perspective and orthographic projection
      (note the D key also toggles between these projections)      
**Clipper**
      Sliders controlling a clipping plane that clips away some of the geometry
      in clipping mode, which is toggled by pressing the **C** key. 
**Trackball**
      Sliders controlling the virtual trackball available to control the viewpoint in 
      modes such as the Rotate mode, which is toggled by pressing the **R** key.
**Bookmarks**
      **Under Development**
**State**
      **Under Development**
**Photon Flag Selection**
      Can be used to select photons based on final flags.
      NB this is different from the photon record history flag selection. 
**GMaterialLib**
      Lists the materials of the geometry, together with their code numbers and colors
**GSurfaceLib**
      Lists the surfaces of the geometry, together with their code numbers and colors
**GFlags**
      Lists the history flags of the propagation, such as **BOUNDARY_TRANSMIT** together
      with code numbers and colors
**Dev**
      Used for GUI development 

 
Some menu sections only appear when propagating photons or with a propagation loaded. Namely:

**Photon Termination Boundaries Selection** 
     **Under development**

**Photon Flag Sequence Selection**
     Checkbox list of photon history sequences, allowing all photons of particular histories to
     be visualized.

**Photon Material Sequence Selection**
     Checkbox list of photon material sequences, allowing all photons of particular material histories to
     be visualized.
     



Navigating in a Geometry
--------------------------

Navigation keys::

     R: rotate mode toggle        drag around rotate around viewpoint 
     S: screen scale mode toggle  drag up/down to change screen scale 
     X: pan mode toggle           drag to move viewpoint within screen plane
     Y: yfov mode toggle 
     Z: zoom mode toggle          (actually changes z position, not zoom) 


Pressing the up-arrow/down-arrow keys doubles/halves the sensitivity of 
all dragging. 8x or 16x sensitivity is useful within larger geometries. 


Bookmarks and Making Interpolated View Animation
----------------------------------------------------

::

     T: Composition::nextViewMode 
     U: Composition::nextViewType, use to cycle thru view/altview : altview is InterpolatedView  

     0-9: jump to preexisting bookmark  
     0-9 + shift: create or update bookmark  
     SPACE: update the current bookmark, commiting trackballing into the view and persisting 


After bookmarking several viewpoints, pressing the U key will interpolate the viewpoint 
from bookmark to bookmark.  Bookmarks are persisted between invokations.
Bookmarks are an area that needs some debugging.  


Controlling Animations and Photon Presentation
-----------------------------------------------

::


     M: Composition::nextColorStyle      m1/m2/f1/f2/p1/p2 
     P: Scene::nextPhotonStyle       dot/longline/shortline
     G: gui mode                        toggle GUI 
 
     Animator modes are changed by pressing keys A,V,T
     
     A: event propagation 
     V: geometry rotation 
     T: interpolated navigation 
     
     Holding SHIFT with A,V,T reverses animation time direction 
     Holding OPTION with A,V,T changes to previous animation mode, instead of next  
     Holding CONTROL with A,V,T sets animation mode to OFF  


**A**
     starts an event propagation animation

**G**
     switches between GUI modes: no-gui/time-scrubber/full-gui-menu

**M**
     controls color of photon representation, based on materials m1/m2 or
     flags f1/f2 or the polarization of the photon p1/p2
      
**P**
     controls the representation of the photon, either a point, a long line
     over the full path of the photon or a shortline indicating the direction
     of the photon.
     In **Composition** GUI section there is a **pick.x** selection that modulo
     selects the photons to display.
       





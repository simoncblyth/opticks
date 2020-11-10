Visualization of Geometry and Event data
==========================================


.. contents:: Table of Contents
   :depth: 2


Launching Visualization 
-------------------------------

Visualization::

   OTracerTest  # just viz, fast start as does no simulation
   OKTest       # does Opticks simulation before viz
   OKG4Test     # does both Geant4 and Opticks simulations before viz


Controlling the initial viewpoint 
-----------------------------------

::

    OTracerTest ---target 3153 --eye -1,-1,-1 --look 0,0,0 --up 0,0,1  
    

target
     node index of target volume, the below eye/look/up options are all relative to this target volume 
eye
     point in extent of the volume units  
look
     point in extent of the volume units
up
     up direction vector 


Use the "--dbgaim" option for extra logging about geometry volumes and targetting.  


Configuring the default target volume 
--------------------------------------

Geant4 auxiliary metadata (G4GDMLAux) on logical volumes can be 
used to configure the default target volume.
This can be done at geometry creation in C++ or at 
GDML level. 

The default geocache geometry corresponds to the 
GDML path returned by opticksaux-dx1.  This GDML 
has the auxiliary element shown below with "label" auxtype and 
"target" auxvalue.  The first placement of this logical volume is thus 
configured as the default target.

::

    epsilon:~ blyth$ grep -b2 target $(opticksaux-dx1) 
    985562-    </volume>
    985576-    <volume name="/dd/Geometry/AD/lvADE0xc2a78c00x3ef9140">
    985636:      <auxiliary auxtype="label" auxvalue="target"/>
    985689-      <materialref ref="/dd/Materials/IwsWater0x3e6bf60"/>
    985748-      <solidref ref="ade0xc2a74380x3eafdb0"/>


See Opticks::getGDMLAuxTargetLVNames GGeo::getFirstNodeIndexForGDMLAuxTargetLVName.
The Geant4 auxiliary setting can be overridden using the OPTICKS_TARGET envvar 
or the "--target 3153" commandline option.::

   export OPTICKS_TARGET=3153 
      ## as the target node index will depend on the geometry   
      ## setting this next to the OPTICKS_KEY is natural



Essential keys to make geometry visible
-------------------------------------------

Often on launching visualization you will get a blank screen.
To make the geometry visible, some things to try:

* press Q twice, this toggles the global (non-instanced) geometry on/off 
* press I several times, this toggles the presentation of instanced geometry  
* press B several times, this switches the render style of the instanced geometry 
* press V, this toggles animated rotation of the geometry 
* press A, this starts the optical photon propagation animation 
* press O, switches between OpenGL rasterized and OptiX ray trace rendering  

If after all that you still have a black screen, try:

* changing the target to a volume appropriate to your geometry, 
  if the target is defaulted to 0 corresponding to the world volume
  and you have a very large world volume your detector might not 
  not be visible.

* press G, this brings up a GUI with help menus etc..


Further Visualization Interaction Tips using "InteractorKeys"
--------------------------------------------------------------

The "InteractorKeys" executable dumps the full list of keys::

    epsilon:~ blyth$ InteractorKeys

     A: Composition::nextMode     record animation, enable and control speed  
     B: Scene::nextGeometryStyle  bbox/norm/wire 
     C: Clipper::next             toggle geometry clipping 
     D: Camera::nextStyle         perspective/orthographic 
     E: Composition::nextGeometryStyle  default(lightshader)/nrmcol/vtxcol/facecol 
     F: far mode toggle : swipe up/down change frustum far 
     G: gui mode    toggle GUI 
     H: Trackball::home  
     I: Scene::nextInstanceStyle style of instanced geometry eg PMT visibility  
     J: Scene::jump  
     K: Composition::nextPickPhotonStyle OR toggle scrub mode 
     L: Composition::nextNormalStyle     flip normal in shaders 
     M: Composition::nextColorStyle      m1/m2/f1/f2/p1/p2  (window title shows eg col:flag2) 
     N: near mode toggle : swipe up/down to change frustum near   
     O: OptiX render mode           raytrace/hybrid/OpenGL 
     P: Scene::nextPhotonStyle       dot/longline/shortline  
     Q: Scene::nextGlobalStyle      non-instanced geometry style: default/normalvec/none 
     R: rotate mode toggle  drag around rotate around viewpoint  
     S: screen scale mode toggle  drag up/down to change screen scale (use in Orthographic)  
     T: Composition::nextViewMode, has effect only with non-standard views (Interpolated, Track, Orbital)
        typically changing animation speed  
     U: Composition::nextViewType, use to toggle between standard and altview : altview mode can be changed with T InterpolatedView   
     V: View::nextMode      rotate view, with shift modifier rotates in opposite direction     
     W: decrease(increase with shift modifier) OptiX rendering resolution by multiples of 2, up to 16x
     X: pan mode toggle 
     Y: yfov mode toggle 
     Z: zoom mode toggle   (actually changes z position, not zoom)  

     ...


These tips are also visible within the onscreen GUI by pressing "G" several 
types and tapping the "Interactor" disclosure triangle.

Particular useful commands for making geometry visible are::

    V: View::nextMode                  rotate view, with shift modifier rotates in opposite direction 
    L: Composition::nextNormalStyle    flip normal in shaders 
    B: Scene::nextGeometryStyle        bbox/norm/wire 
    C: Clipper::next                   toggle geometry clipping 
    E: Composition::nextGeometryStyle  lightshader/normalshader/flatvertex/facecolor 
    D: Camera::nextStyle               perspective/orthographic 
    O: OptiX render mode               raytrace/hybrid/OpenGL
    G: gui mode                        toggle GUI 


**L**
     flips normals so light from outside a volume can still light inside the volume 
**E** 
     changes the coloring style



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
       





Visualization of Geometry and Event data
==========================================

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
     flips normals so light from outside a volume can light inside it 
**E** 
     changes the coloring style


For a full list of keys::

    op --keys  


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


Controlling Animations
------------------------

::

     Animator modes are changed by pressing keys A,V,T
     
     A: event propagation 
     V: geometry rotation 
     T: interpolated navigation 
     
     Holding SHIFT with A,V,T reverses animation time direction 
     Holding OPTION with A,V,T changes to previous animation mode, instead of next  
     Holding CONTROL with A,V,T sets animation mode to OFF  






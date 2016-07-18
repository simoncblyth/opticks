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
    R: rotate mode                     drag around rotate around viewpoint 
    Y: yfov mode                       drag up/down to change field of view
    G: gui mode                        toggle GUI 


Particularly useful commands for making geometry visible are:

**L**
     flips normals so light from outside a volume can light inside it 
**E** 
     changes the coloring style


For a full list of keys::

    op --keys  




Leaky Analytic Sphere
========================

Executive Summary
------------------

Avoid doing tests that involve shooting millions of photons at points 
where bounding boxes touch geometry, 
OR enlarge bounding boxes by very small factors to avoid the touching.::

    debug=1.000001


Refs about floating point epsilon
----------------------------------

* https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/


Polar Leak
------------

::

    local torch_config=(
                 type=refltest
                 photons=500000
                 polz=${pol}pol
                 frame=1
                 source=0,0,300
                 target=0,0,0
                 radius=100
                 zenithazimuth=0,0.5,0,1
                 material=Vacuum
               )

    local test_config=(
                 mode=BoxInBox
                 dimensions=500,300,0,0
                 shape=S,S
                 boundary=Rock//perfectAbsorbSurface/Vacuum
                 boundary=Vacuum///Pyrex 
                 analytic=1
               )


Shooting millions of photons from Vacuum at the +Z pole of Pyrex sphere  
from a hemi-spherical torch source leads some 8% incorrectly 
thinking already in pyrex when actually in vacuum 
(ie rays failed to intersect at the targetted +Z pole outer surface instead intersecting 
all the way across on other side of sphere from "inside")


Select them interactively with material sequences starting Py::

    [2015-Nov-17 11:39:11.677236]:info: 
        3     24686     0.049                      dde                               Py Vm Vm 
        5     10752     0.022                       ee                                  Py Py 
       10       979     0.002                     ddee                            Py Py Vm Vm 


This is visible as a red Pyrex volcanic plume within the mostly white 
outside the sphere.  The plume is centered around the normal direction but is 
fairly broad.

The targetted point is special as thats along axis where the axis aligned bounding box 
touches the sphere.


Scaling the BBox
-----------------

Is the bbox effectively clipping the sphere leaving a hole for the photons to sneak thru ?
YEP, seems so : scaling the bbox my a very small amount eliminates the problem,

*  *zero/0.5M* start with the wrong material Py 

::

    debug=1.000001



Radial Shooting from all directions
-------------------------------------

Two concentic spheres of radii 500 and 300 mm with spherical
torch of radius 400 mm with emitting in inwards radial direction 
(source=0,0,0). 

No signs of any leaking here, so not an issue with normal incidence.

::

    local torch_config=(
                 type=refltest
                 photons=500000
                 polz=${pol}pol
                 frame=1
                 source=0,0,0
                 target=0,0,0
                 radius=400
                 zenithazimuth=0,0.5,0,1
                 material=Vacuum
               )

    local test_config=(
                 mode=BoxInBox
                 dimensions=500,300,0,0
                 shape=S,S
                 boundary=Rock//perfectAbsorbSurface/Vacuum
                 boundary=Vacuum///Pyrex 
                 analytic=1
               )



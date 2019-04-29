torus_intersect_alternative_with_raymarching_aka_sphere_tracing
==================================================================



* :google:`Andrew Kensler` 





Matt Zucker
-------------

* https://github.com/mzucker?tab=repositories
* https://github.com/mzucker/miniray
* https://mzucker.github.io/2016/08/03/miniray.html

One immediate obstacle arose: the result of sweeping a 3D sphere along a 2D arc
is a torus segment, and ray-torus intersection a bit hairy because it requires
solving a quartic (fourth-order polynomial). Analytic solutions to general
quartics exist, but they’re complex enough to spill the program way over the
size of a business card.

Fortunately, I had recently come across Inigo Quilez’s nvscene 2008
presentation, Rendering Worlds With Two Triangles, which supplies a convenient
dodge: raymarching, a.k.a. sphere tracing. Here’s an illustration of the
process, from iq’s slides:

Sphere tracing works by computing or estimating the distance between the
current point on the ray (initially the ray origin) and the nearest point in
the scene, and advancing along the ray by that distance, as shown above.
Although it may be slower than analytic raytracying, it is numerically robust
and guaranteed to converge to an intersection as long as the ray starts in free
space and the distance estimate is less than or equal to the true distance.

While analytically intersecting a ray and torus is hairy, computing the exact
distance between a point and a torus is a piece of cake, and it admits a
straightforward solution to clamping the angle to obtain arcs of less than
360 deg.

Distance between a point and a torus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

::

    float sdTorus( vec3 p, vec2 t )
    {
      vec2 q = vec2(length(p.xz)-t.x,p.y);
      return length(q)-t.y;
    }





Inigo Quilez : Ray marching, rendering with distance fields
----------------------------------------------------------------

* http://iquilezles.org/www/material/nvscene2008/nvscene2008.htm
* http://iquilezles.org/www/material/nvscene2008/rwwtt.pdf

Raymarching : Kind of raytracing for all those objects that do not
have an analytic intersection function.

Similarly previous works

* "Ray tracing deterministic 3-D fractals" published at Siggraph 1989 by D.J.Sandin and others.
* "Per-pixel displacement mapping with distance functions", appeared in GPU Gems 2 (2005) by W.Donnelly.

The trick is to be able to compute or estimate (a lower bound of) the distance to the 
closest surface at any point in space. 

* This allows for marching in large steps along the ray


Ray  Marching
~~~~~~~~~~~~~~~~~

* http://iquilezles.org/www/articles/raymarchingdf/raymarchingdf.htm
* https://github.com/search?q=raymarching



Jamie Wong
---------------

* http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/


In raymarching, the entire scene is defined in terms of a signed distance
function. To find the intersection between the view ray and the scene, we start
at the camera, and move a point along the view ray, bit by bit. At each step,
we ask “Is this point inside the scene surface?”, or alternately phrased, “Does
the SDF evaluate to a negative number at this point?“. If it does, we’re done!
We hit something. If it’s not, we keep going up to some maximum number of steps
along the ray.

We could just step along a very small increment of the view ray every time, but
we can do much better than this (both in terms of speed and in terms of
accuracy) using “sphere tracing”. Instead of taking a tiny step, **we take the
maximum step we know is safe without going through the surface: we step by the
distance to the surface, which the SDF provides us!**



Handling non-uniform scaling (eg ellipsoid) 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/

The principle for other non-rigid transformations is the same: as long as the
sign is preserved by the transformation, you just need to figure out some
compensation factor to ensure that you’re never overestimating the distance to
the surface.


Celarek
-----------

* http://celarek.at/tag/ray-marching/
* http://celarek.at/wp/wp-content/uploads/2014/05/realTimeFractalsReport.pdf


Ray Marching Sphere Tracing
----------------------------

* :google:`ray marching sphere tracing`



bisection refinement
~~~~~~~~~~~~~~~~~~~~~~

* https://www.scratchapixel.com/lessons/advanced-rendering/rendering-distance-fields

::


    // bisection method
    // we assume evaluateImplicitFunction(P4) > 0 and evaluateImplicitFunction(P5) < 0
    float threshold = 0.0001; 
    Vec3f intersectP; 
    while (1) { 
        Vec3f midPoint = (P4 + P5) * 0.5; 
        float signedDistance = evaluateImplicitFunction(midPoint); 
        if (fabs(signedDistance) < threshold) { 
            intersectP = midPoint; // this point is close enough to the surface 
            break; 
        } 
        // new segment
        if (signedDistance < 0) { 
            P5 = midPoint; 
        } 
        else { 
            P4 = midPoint; 
        } 
    } 


Sphere tracing: a geometric method for the antialiased ray tracing of implicit surfaces, John C. Hart, 1996.

* http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.3825&rep=rep1&type=pdf



Shadertoy interactive example of ray marching : took minutes to load into browser
----------------------------------------------------------------------------------

* https://www.shadertoy.com/view/XsB3Rm

Added Torus::

    float sdTorus(vec3 p, vec2 t)
    {
        vec2 q = vec2(length(p.xz)-t.x,p.y);
        return length(q) - t.y ;  
    }


With distance field::

    const vec2 t = vec2(0.5, 0.1) ; 
    float d = sdTorus( p, t ); 



* https://www.shadertoy.com/view/Xds3zN


Ray Marching : an approximate approach to find intersect with distance field geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.shadertoy.com/view/XsB3Rm

* once have an approximate intersect : can then make it "effectively" exact

  * "refinement" in below code is a starting point for this
  * iterate across the surface in smaller and smaller steps 
    until they are small enough for the desired accuracy 

Code from shadertoy demo::

    bool ray_marching( vec3 o, vec3 dir, inout float depth, inout vec3 n){

        float t = 0.0 ; 
        float d = 10000.0 ; 
        float dt = 0.0 ; 
        for( int i= 0 ; i < 128 ; i++)
        {
            vec3 v = o + dir*t ;   // start at o
            d = dist_field( v );   // closest distance to surface in any direction
            if( d < 0.001 ){
                break ; 
            }
            dt = min( abs(d), 0.1 ) ;   // absolute distance but not smaller than 0.1 : always +ve so cannot go backwards 
            t += dt ; 
            if( t > depth )
            {
                break ;  
            }
        }

        if( d >= 0.001 ){
            return false ; 
        } 
     
        t -= dt ; 
        for( int i=0 ; i < 4 ; i++)
        {
            dt *= 0.5 ;        // refinement
            vec3 v = o + dir*( t+dt ); 
            if( dist_field(v) >= 0.001 )
            {
                t += dt ; 
            }  
        }

        depth = t ; 
        n = normalize( gradient( o+dir*t ));
        return true ; 
    }





iq : ray marching primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.shadertoy.com/view/Xds3zN




#pragma once

#include "SPlaceSphere.h"
#include "SPlaceCylinder.h"
#include "SPlaceCircle.h"


struct SPlace
{
    static constexpr const char* OPTS = "TR,tr,R,T,r,t" ; 
    static NP* AroundSphere(   const char* opts, double radius, double item_arclen, unsigned num_ring=10 ); 
    static NP* AroundCylinder( const char* opts, double radius, double halfheight , unsigned num_ring=10, unsigned num_in_ring=16  ); 
    static NP* AroundCircle(   const char* opts, double radius, unsigned num_in_ring=4  ); 
};


/**
SPlace::AroundSphere
-----------------------

The number within each ring at a particular z is determined 
from the circumference the ring at that z. 





            +---+            
        +------------+
      +----------------+
        +------------+
            +---+            


     x = r sin th cos ph
     y = r sin th sin ph       x^2 + y^2 = r^2 sin^2 th
     z = r cos th                    z^2 = r^2 cos^2 th
              
    cos th = z/r 
    sin th = sqrt( 1 - (z/r)^2 )


               r sin th
         +. .+. .+
          \  |  /|
           \ | / | r cos th
            \|/  |
   ---+------+---+---+ 
                 r


Circumference of ring : 2*pi*(r*sin th) = 2*pi*r*sqrt(1 - (z/r)^2 )     
at z=0 -> 2*pi*r at z =r  -> 0  

**/


NP* SPlace::AroundSphere(const char* opts_, double radius, double item_arclen, unsigned num_ring )
{
    const char* opts = opts_ ? opts_ : OPTS ; 
    SPlaceSphere sp(radius, item_arclen, num_ring); 
    return sp.transforms(opts)  ; 
}


/**
SPlace::AroundCylinder
------------------------

Form placement transforms that orient local-Z axis to 
a radial outwards direction with translation at points around the cylinder.  

flip:false(default)
    transform memory layout has last 4 of 16 elements (actually 12,13,14) 
    holding the translation, which corresponds to the OpenGL standard

flip:true
    transform memory layout has translation in right hand column at elements 3,7,11
    this is needed by pyvista it seems 
    This corresponds to the transposed transform compared with flip:false

**/

NP* SPlace::AroundCylinder(const char* opts_, double radius, double halfheight, unsigned num_ring, unsigned num_in_ring )
{
    const char* opts = opts_ ? opts_ : OPTS ; 
    SPlaceCylinder cy(radius, halfheight, num_ring, num_in_ring ); 
    return cy.transforms(opts)  ; 
}


/**
SPlace::AroundCircle
---------------------

Circle currently fixed in XZ plane 

**/

NP* SPlace::AroundCircle(const char* opts_, double radius, unsigned num_in_ring )
{
    const char* opts = opts_ ? opts_ : OPTS ; 
    SPlaceCircle ci(radius, num_in_ring ); 
    return ci.transforms(opts)  ; 
}



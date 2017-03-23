
#include "NBBox.hpp"
#include "NBox.hpp"
#include "NPart.hpp"
#include "NPlane.hpp"

#include <cmath>
#include <cassert>
#include <cstring>

#include "OpticksCSG.h"

// signed distance function

double nbox::operator()(double px, double py, double pz) 
{
    nvec3 p = make_nvec3( px - param.x, py - param.y, pz - param.z ); // in the frame of the box
    nvec3 a = nabsf(p) ; 
    nvec3 s = make_nvec3( param.w, param.w, param.w );          
    nvec3 d = a - s ; 
    return nmaxf(d) ;
} 

/**
    ~/opticks_refs/Procedural_Modelling_with_Signed_Distance_Functions_Thesis.pdf

    SDF from point px,py,pz to box at origin with side lengths (sx,sy,sz) at the origin 

    max( abs(px) - sx/2, abs(py) - sy/2, abs(pz) - sz/2 )


**/

nbbox nbox::bbox()
{
    nbbox bb ;
    float s  = param.w ; 
    bb.min = make_nvec3( param.x - s, param.y - s, param.z - s );
    bb.max = make_nvec3( param.x + s, param.y + s, param.z + s );
    return bb ; 
}



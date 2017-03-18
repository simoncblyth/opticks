
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
    double s = param.w ; 
    return fmax(fmax(fabs(px)-s,fabs(py)-s),fabs(pz)-s) ; 
} 

void nbox::dump(const char*)
{
    param.dump("nbox");
}

npart nbox::part()
{
    float s  = param.w ; 

    nbbox bb ;
    bb.min = make_nvec4( param.x - s, param.y - s, param.z - s, 0.f );
    bb.max = make_nvec4( param.x + s, param.y + s, param.z + s, 0.f );

    npart p ; 
    p.zero();            
    p.setParam(param) ; 
    p.setTypeCode(CSG_BOX); 
    p.setBBox(bb);

    return p ; 
}



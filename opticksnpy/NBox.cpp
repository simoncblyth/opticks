
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
    double s = param.w ; 
    return fmax(fmax(fabs(px)-s,fabs(py)-s),fabs(pz)-s) ; 
} 

void nbox::dump(const char*)
{
    param.dump("nbox");
}

nbbox nbox::bbox()
{
    nbbox bb ;
    float s  = param.w ; 
    bb.min = make_nvec3( param.x - s, param.y - s, param.z - s );
    bb.max = make_nvec3( param.x + s, param.y + s, param.z + s );
    return bb ; 
}

npart nbox::part()
{
    nbbox bb = bbox();

    npart p ; 
    p.zero();            
    p.setParam(param) ; 
    p.setTypeCode(CSG_BOX); 
    p.setBBox(bb);

    return p ; 
}



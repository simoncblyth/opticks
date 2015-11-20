#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include <cmath>

ndisc nsphere::intersect(nsphere& a, nsphere& b)
{
    // Find Z intersection disc of two Z offset spheres,
    // disc radius is set to zero when no intersection.
    //
    // http://mathworld.wolfram.com/Circle-CircleIntersection.html
    //
    // cf pmt-/dd.py 

    float R = a.param.w ; 
    float r = b.param.w ; 

    // operate in frame of Sphere a 

    float dx = b.param.x - a.param.x ; 
    float dy = b.param.y - a.param.y ; 
    float dz = b.param.z - a.param.z ; 

    assert(dx == 0 && dy == 0 && dz != 0);

    float d = dz ;  
    float dd_m_rr_p_RR = d*d - r*r + R*R  ; 
    float z = dd_m_rr_p_RR/(2.*d) ;
    float yy = (4.*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.*d*d)  ;
    float y = yy > 0 ? sqrt(yy) : 0 ;   

    return ndisc(nplane(0,0,1,z + a.param.z),y);  // return to original frame
}


void nsphere::dump(const char* msg)
{
    param.dump(msg);
}


npart nsphere::part()
{
    //npart p = npart();   //  ctor like this zeroes
    npart p ; 
    p.zero();              // but this is clearer

    p.q0.f = param ; 

    float z = param.z ;  
    float radius = param.w ; 

    nbbox bb(z-radius, z+radius, -radius, radius);

    p.q2.f = bb.min ; 
    p.q3.f = bb.max ; 

    p.setTypeCode(SPHERE); // must come after setting bbox 

    return p ; 
}



npart nsphere::zlhs(float z)
{
    npart p = part();
    return p ; 
}


npart nsphere::zrhs(float z)
{
    npart p = part();
    return p ; 
}


#include "NSphere.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"

#include <cmath>
#include <cassert>

ndisc nsphere::intersect(nsphere& a, nsphere& b)
{
    // Find Z intersection disc of two Z offset spheres,
    // disc radius is set to zero when no intersection.
    //
    // http://mathworld.wolfram.com/Circle-CircleIntersection.html
    //
    // cf pmt-/dd.py 

    float R = a.radius() ; 
    float r = b.radius() ; 

    // operate in frame of Sphere a 

    float dx = b.x() - a.x() ; 
    float dy = b.y() - a.y() ; 
    float dz = b.z() - a.z() ; 

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
    float _z = z() ;  
    float r  = radius() ; 

    nbbox bb(_z - r, _z + r, -r, r);

    npart p ; 
    p.zero();            
    p.setParam(param) ; 
    p.setTypeCode(SPHERE); 
    p.setBBox(bb);

    return p ; 
}



npart nsphere::zlhs(const ndisc& dsk)
{
    npart p = part();

    float _z = z() ;  
    float r  = radius() ; 
    nbbox bb(_z - r, dsk.z(), -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}

npart nsphere::zrhs(const ndisc& dsk)
{
    npart p = part();

    float _z = z() ;  
    float r  = radius() ; 
    nbbox bb(dsk.z(), _z + r, -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}


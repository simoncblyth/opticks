
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>


#include "NGLMExt.hpp"

// sysrap-
#include "OpticksCSG.h"

// npy-
#include "NSphere.hpp"
#include "NBBox.hpp"
#include "NPlane.hpp"
#include "NPart.hpp"
#include "Nuv.hpp"

#include "PLOG.hh"


float nsphere::costheta(float z)
{
   return (z - center.z)/radius ;  
}

// signed distance function

float nsphere::operator()(float x, float y, float z) const 
{
    glm::vec4 p(x,y,z,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform
    float sd = glm::distance( glm::vec3(p), center ) - radius ;
    return complement ? -sd : sd ;
} 


unsigned nsphere::par_nsurf() const 
{
   return 1 ; 
}
int nsphere::par_euler() const 
{
   return 2 ; 
}
unsigned nsphere::par_nvertices(unsigned nu, unsigned nv) const 
{
   // expected unique vertex count, accounting for extra ones, poles and 360-seam 
   assert( nv > 2 ); 
   return 2 + (nu+1-1)*(nv+1-2) ;     
}


glm::vec3 nsphere::par_pos(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    unsigned u  = uv.u() ; 
    unsigned v  = uv.v() ; 
    unsigned nu = uv.nu() ; 
    unsigned nv = uv.nv() ; 

    assert(s < par_nsurf());

    // Avoid numerical precision problems at the poles
    // by providing precisely the same positions
    // and on the 360 degree seam by using 0 degrees at 360 
    
    bool is_north_pole = v == 0 ; 
    bool is_south_pole = v == nv ; 
    bool is_360_seam = u == nu ; 

    float fu = is_360_seam ? 0.f : float(u)/float(nu) ; 
    float fv = float(v)/float(nv) ; 

    const float pi = glm::pi<float>() ;
    float azimuth = fu * 2 * pi ;
    float polar   = fv * pi ;


    glm::vec3 pos(center) ;  
    if(is_north_pole || is_south_pole)
    {
        pos += glm::vec3(0,0,is_north_pole ? radius : -radius ) ; 
    }   
    else
    { 
        float ca = cosf(azimuth);
        float sa = sinf(azimuth);
        float cp = cosf(polar);
        float sp = sinf(polar);

        pos += glm::vec3( radius*ca*sp, radius*sa*sp, radius*cp );
    }



    /*
    std::cout << "nsphere::par_pos"
              << " u " << std::setw(3) << u 
              << " v " << std::setw(3) << v
              << " nu " << std::setw(3) << nu 
              << " nv " << std::setw(3) << nv
              << " azimuth " << std::setw(15) << std::fixed << std::setprecision(4) << azimuth
              << " polar " << std::setw(15) << std::fixed << std::setprecision(4) << polar
              << " pos "
              << " " << std::setw(15) << std::fixed << std::setprecision(4) << pos.x
              << " " << std::setw(15) << std::fixed << std::setprecision(4) << pos.y
              << " " << std::setw(15) << std::fixed << std::setprecision(4) << pos.z
              << " " << ( is_north_pole ? "north_pole" : "" )
              << " " << ( is_south_pole ? "south_pole" : "" )
              << " " << ( is_360_seam ? "360_seam" : "" )
              << std::endl 
              ;
     */
   

    return pos ; 
}







glm::vec3 nsphere::gseedcenter()
{
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ; // t:transform
}


void nsphere::pdump(const char* msg, int verbosity)
{
    nnode::dump();
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center 
              << " radius " << radius 
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}



nbbox nsphere::bbox() const 
{
    nbbox bb = make_bbox();

    bb.min = make_nvec3(center.x - radius, center.y - radius, center.z - radius);
    bb.max = make_nvec3(center.x + radius, center.y + radius, center.z + radius);
    bb.side = bb.max - bb.min ; 
    bb.invert = complement ; 
    bb.empty = false ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 
}



void nsphere::adjustToFit(const nbbox& bb, float scale)
{
    nquad qce ; 
    qce.f = bb.center_extent() ; 
    qce.f.w *= scale ; 

    init_sphere( *this, qce );
}








ndisc nsphere::intersect(nsphere& a, nsphere& b)
{
    // Find Z intersection disc of two Z offset spheres,
    // disc radius is set to zero when no intersection.
    //
    // http://mathworld.wolfram.com/Circle-CircleIntersection.html
    //
    // cf pmt-/ddpart.py 

    float R = a.radius ;
    float r = b.radius ; 

    // operate in frame of Sphere a 

    float dx = b.center.x - a.center.x ; 
    float dy = b.center.y - a.center.y ; 
    float dz = b.center.z - a.center.z ; 

    assert(dx == 0 && dy == 0 && dz != 0);

    float d = dz ;  
    float dd_m_rr_p_RR = d*d - r*r + R*R  ; 
    float z = dd_m_rr_p_RR/(2.f*d) ;
    float yy = (4.f*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.f*d*d)  ;
    float y = yy > 0 ? sqrt(yy) : 0 ;   


    nplane plane = make_plane(0,0,1,z + a.center.z) ;
    ndisc  disc = make_disc(plane, y) ;

    return disc ;      // return to original frame
}


npart nsphere::part()
{
    npart p = nnode::part();

    assert( p.getTypeCode() == CSG_SPHERE );

    if(npart::VERSION == 0u)
    {
        // TODO: move this belongs into a zsphere not here ???
        LOG(warning) << "nsphere::part override bbox " ;  
        float z = center.z ;  
        float r  = radius ; 
        nbbox bb = make_bbox(z - r, z + r, r, r);

        p.setBBox(bb);
    }
    return p ; 
}



npart nsphere::zlhs(const ndisc& dsk)
{
    npart p = part();

    float z = center.z ;  
    float r  = radius ; 
    nbbox bb = make_bbox(z - r, dsk.z(), -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}

npart nsphere::zrhs(const ndisc& dsk)
{
    npart p = part();

    float z = center.z ;  
    float r  = radius ; 
    nbbox bb = make_bbox(dsk.z(), z + r, -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}



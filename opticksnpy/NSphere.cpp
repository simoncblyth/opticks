
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


float nsphere::costheta(float z_)
{
   return (z_ - z())/radius() ;  
}

// signed distance function

float nsphere::operator()(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.f); 
    if(gtransform) p = gtransform->v * p ;  // v:inverse-transform
    glm::vec3 c = center(); 
    float sd = glm::distance( glm::vec3(p), c ) - radius() ;
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


glm::vec3 nsphere::par_pos_model(const nuv& uv) const 
{
    unsigned s  = uv.s(); 
    assert(s == 0);

    glm::vec3 c = center();
    glm::vec3 pos(c);

    float r_ = radius();

    _par_pos_body( pos, uv, r_ );
  
    return pos ; 
}





void nsphere::par_posnrm_model( glm::vec3& pos, glm::vec3& nrm, unsigned s, float fu, float fv ) const 
{ 
    assert(s == 0);

    glm::vec3 c = center();

    pos.x = c.x ; 
    pos.y = c.y ; 
    pos.z = c.z ; 

    nrm.x = 0.f ; 
    nrm.y = 0.f ; 
    nrm.z = 0.f ; 

    float r_ = radius();
    bool is_north_pole = fv == 0.f ;  // highly unlikely for dice to land on the poles
    bool is_south_pole = fv == 1.f ; 

    if(is_north_pole || is_south_pole) 
    {
        pos += glm::vec3(0,0,is_north_pole ? r_ : -r_ ) ; 
        nrm.z = is_north_pole ? 1.f : -1.f ;         
    }   
    else
    {
        const float pi = glm::pi<float>() ;
        float azimuth = 2.f*pi*fu ; 
        float polar = pi*fu ; 

        float ca = cosf(azimuth);
        float sa = sinf(azimuth);
        float cp = cosf(polar);
        float sp = sinf(polar);

        pos += glm::vec3( r_*ca*sp, r_*sa*sp, r_*cp );
 
        nrm.x = ca*sp ; 
        nrm.y = sa*sp ; 
        nrm.z = cp ; 
    } 
}


void nsphere::_par_pos_body(glm::vec3& pos,  const nuv& uv, const float r_ )  // static
{
    unsigned  v  = uv.v(); 
    unsigned nv  = uv.nv(); 

    // Avoid numerical precision problems at the poles
    // by providing precisely the same positions
    // and on the 360 degree seam by using 0 degrees at 360 
    
    bool is_north_pole = v == 0 ; 
    bool is_south_pole = v == nv ; 

    if(is_north_pole || is_south_pole) 
    {
        pos += glm::vec3(0,0,is_north_pole ? r_ : -r_ ) ; 
    }   
    else
    { 
        bool seamed = true ; 

        float azimuth = uv.fu2pi(seamed); 
        float polar = uv.fvpi() ; 

        float ca = cosf(azimuth);
        float sa = sinf(azimuth);
        float cp = cosf(polar);
        float sp = sinf(polar);

        pos += glm::vec3( r_*ca*sp, r_*sa*sp, r_*cp );
    }
}






glm::vec3 nsphere::gseedcenter() const 
{
    glm::vec3 c = center();
    return gtransform == NULL ? c : glm::vec3( gtransform->t * glm::vec4(c, 1.f ) ) ; // t:transform
}


void nsphere::pdump(const char* msg) const 
{
/*
Avoid some unintended recursion here

 frame #24077: 0x0000000100b7a3c3 libNPY.dylib`NNodeDump::dump_prim(this=0x0000000102d03c50) const + 531 at NNodeDump.cpp:125
    frame #24078: 0x0000000100b79d3f libNPY.dylib`NNodeDump::dump(this=0x0000000102d03c50) const + 63 at NNodeDump.cpp:27
    frame #24079: 0x0000000100b557ce libNPY.dylib`nnode::dump(this=0x00007fff5fbfec40, msg=0x0000000000000000) const + 270 at NNode.cpp:1098
    frame #24080: 0x0000000100ba75f3 libNPY.dylib`nsphere::pdump(this=0x00007fff5fbfec40, msg=0x0000000100d6d78d) const + 51 at NSphere.cpp:114
    frame #24081: 0x0000000100b7a3c3 libNPY.dylib`NNodeDump::dump_prim(this=0x0000000102d03c50) const + 531 at NNodeDump.cpp:125
    frame #24082: 0x0000000100b79d3f libNPY.dylib`NNodeDump::dump(this=0x0000000102d03c50) const + 63 at NNodeDump.cpp:27
    frame #24083: 0x0000000100b557ce libNPY.dylib`nnode::dump(this=0x00007fff5fbfec40, msg=0x0000000000000000) const + 270 at NNode.cpp:1098
    frame #24084: 0x0000000100ba75f3 libNPY.dylib`nsphere::pdump(this=0x00007fff5fbfec40, msg=0x0000000100d6d78d) const + 51 at NSphere.cpp:114
    frame #24085: 0x0000000100b7a3c3 libNPY.dylib`NNodeDump::dump_prim(this=0x0000000102d03c50) const + 531 at NNodeDump.cpp:125
    frame #24086: 0x0000000100b79d3f libNPY.dylib`NNodeDump::dump(this=0x0000000102d03c50) const + 63 at NNodeDump.cpp:27
    frame #24087: 0x0000000100b557ce libNPY.dylib`nnode::dump(this=0x00007fff5fbfec40, msg=0x0000000000000000) const + 270 at NNode.cpp:1098
    frame #24088: 0x0000000100ba75f3 libNPY.dylib`nsphere::pdump(this=0x00007fff5fbfec40, msg=0x00000001000113f3) const + 51 at NSphere.cpp:114
    frame #24089: 0x0000000100005591 NNode2Test`test_generateParPoints_sphere() + 97 at NNode2Test.cc:52
    frame #24090: 0x00000001000057c4 NNode2Test`main(argc=1, argv=0x00007fff5fbfee00) + 468 at NNode2Test.cc:65
    frame #24091: 0x00007fff880d35fd libdyld.dylib`start + 1

*/

   // nnode::dump();  // <-- THIS CAUSES UNINTENDED RECURSION
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center()
              << " radius " << radius()
              << " gseedcenter " << gseedcenter()
              << " gtransform " << !!gtransform 
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}



nbbox nsphere::bbox() const 
{

    float  r = radius(); 
    glm::vec3 c = center();

    glm::vec3 mi(c.x - r, c.y - r, c.z - r);
    glm::vec3 mx(c.x + r, c.y + r, c.z + r);
    nbbox bb = make_bbox(mi, mx, complement);

    return gtransform ? bb.make_transformed(gtransform->t) : bb ; 
}



void nsphere::adjustToFit(const nbbox& bb, float scale)
{
    nquad qce ; 
    qce.f = bb.center_extent() ; 
    qce.f.w *= scale ; 

    init_sphere( *this, qce );
}








ndisk nsphere::intersect(nsphere& a, nsphere& b)
{
    // Find Z intersection disk of two Z offset spheres,
    // disk radius is set to zero when no intersection.
    //
    // http://mathworld.wolfram.com/Circle-CircleIntersection.html
    //
    // cf pmt-/ddpart.py 

    float R = a.radius() ;
    float r = b.radius() ; 

    // operate in frame of Sphere a 

    glm::vec3 a_center = a.center();
    glm::vec3 b_center = b.center();



    float dx = b_center.x - a_center.x ; 
    float dy = b_center.y - a_center.y ; 
    float dz = b_center.z - a_center.z ; 

    assert(dx == 0 && dy == 0 && dz != 0);

    float d = dz ;  
    float dd_m_rr_p_RR = d*d - r*r + R*R  ; 
    float z = dd_m_rr_p_RR/(2.f*d) ;
    float yy = (4.f*d*d*R*R - dd_m_rr_p_RR*dd_m_rr_p_RR)/(4.f*d*d)  ;
    float y = yy > 0 ? sqrt(yy) : 0 ;   


    nplane plane = make_plane(0,0,1,z + a_center.z) ;
    ndisk  disk = make_disk(plane, y) ;

    return disk ;      // return to original frame
}


npart nsphere::part() const 
{
    npart p = nnode::part();

    assert( p.getTypeCode() == CSG_SPHERE );

    if(npart::VERSION == 0u)
    {
        // TODO: move this belongs into a zsphere not here ???
        LOG(warning) << "nsphere::part override bbox " ;  
        float z_ = z() ;  
        float r  = radius() ; 
        nbbox bb = make_bbox_zsymmetric(z_ - r, z_ + r, r, r);

        p.setBBox(bb);
    }
    return p ; 
}



npart nsphere::zlhs(const ndisk& dsk)
{
    npart p = part();

    float z_ = z() ;  
    float r  = radius() ; 
    nbbox bb = make_bbox_zsymmetric(z_ - r, dsk.z(), -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}

npart nsphere::zrhs(const ndisk& dsk)
{
    npart p = part();

    float z_ = z() ;  
    float r  = radius() ; 
    nbbox bb = make_bbox_zsymmetric(dsk.z(), z_ + r, -dsk.radius, dsk.radius);
    p.setBBox(bb);

    return p ; 
}



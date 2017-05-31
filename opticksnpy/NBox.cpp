
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp>

#include "NBBox.hpp"
#include "NBox.hpp"
#include "NPart.hpp"
#include "NPlane.hpp"
#include "Nuv.hpp"

#include <cmath>
#include <cassert>
#include <cstring>

#include "OpticksCSG.h"

/**
~/opticks_refs/Procedural_Modelling_with_Signed_Distance_Functions_Thesis.pdf

SDF from point px,py,pz to box at origin with side lengths (sx,sy,sz) at the origin 

    max( abs(px) - sx/2, abs(py) - sy/2, abs(pz) - sz/2 )

**/

float nbox::operator()(float x, float y, float z) const 
{
    glm::vec4 q(x,y,z,1.0); 
    if(gtransform) q = gtransform->v * q ;

    glm::vec3 p = glm::vec3(q) - center ;  // coordinates in frame with origin at box center 
    glm::vec3 a = glm::abs(p) ;

    float sd = 0.f ; 
    if(is_box3)
    {
        glm::vec3 s( param.f.x/2.f, param.f.y/2.f, param.f.z/2.f );      
        glm::vec3 d = a - s ; 
        sd = glm::compMax(d) ;
    }
    else
    {
        glm::vec3 s( param.f.w );      
        glm::vec3 d = a - s ; 
        sd = glm::compMax(d) ;
    }
    return sd ; 
} 

float nbox::sdf1(float x, float y, float z)  
{
    return (*this)(x,y,z);
}

float nbox::sdf2(float x, float y, float z)
{
    glm::vec4 p(x,y,z,1.0); 
    if(gtransform) p = gtransform->v * p ;

    glm::vec3 bmax = is_box3 ? glm::vec3(param.f.x/2.f, param.f.y/2.f, param.f.z/2.f) : center + glm::vec3( param.f.w ) ; // 

    glm::vec3 d = glm::abs(glm::vec3(p)) - bmax  ;

    float dmaxcomp = glm::compMax(d);

    glm::vec3 dmax = glm::max( d, glm::vec3(0.f) );

    float d_inside = fminf(dmaxcomp, 0.f);
    float d_outside = glm::length( dmax );

    return d_inside + d_outside ;       

   // see tests/NBoxTest.cc   sdf2 and sdf1 match despite code appearances
}






unsigned nbox::par_nsurf() const 
{
   return 6 ; 
}
int nbox::par_euler() const 
{
   return 2 ; 
}
unsigned  nbox::par_nvertices(unsigned nu, unsigned nv) const 
{
   assert( nu >=1 && nv >= 1 ); 

/*

                 +-----------+
                /|          /| 
               / |         / |
              7-----8-----9  |
              |  |        |  |
              |  |        |  |
              4  +--5-----6--+
              | /         | /
              |/          |/
              1-----2-----3
 
                               (nu = 2, nv = 2)    


                (nu+1)*(nv+1) = 3*3 = 9      distinct vertices for one face

            (nu+1-2)*(nv+1-2) = 
                (nu-1)*(nv-1) = 1*1 = 1       mid-vertices (not shared)

                     2*(nv-1) = 2*1 = 2       u edges (shared with one other, so only count half of em)
                     2*(nu-1) = 2*1 = 2       v edges (ditto)

       
     fvert = lambda nu,nv:(nu-1)*(nv-1) + (nv-1) + (nu-1) 
     nvert = lambda nu,nv:8+6*fvert(nu,nv)

*/

   unsigned mid_vert = (nv+1-2)*(nu+1-2) ;
   unsigned edge_vert = (nv+1-2) + (nu+1-2) ;
   unsigned face_vert = mid_vert + edge_vert   ;  
   unsigned distinct_vert = 8 + 6*face_vert ; 

   return distinct_vert ; 
}


glm::vec3 nbox::par_pos(const nuv& uv) const 
{
    /*

                 6-----------7
                /|          /| 
               / |         / |
              4-----------5  |
              |  |        |  |
              |  |        |  |
              |  2--------|--3
              | /         | /
              |/          |/
              0-----------1
          
        z
         |  y
         | /
         |/
         +---> x
        

    0:    (0,2,1) (1,2,3)  :   -z
    1:    (4,5,6) (6,5,7)  :   +z

    2:    (0,4,2) (2,4,6)  :   -x
    3:    (1,3,5) (5,3,7)  :   +x

    4:    (1,5,0) (0,5,4)  :   -y
    5:    (6,7,2) (2,7,3)  :   +y

    */

    nbbox bb = bbox() ;



    glm::vec3 p ; 

    //   1 - uv[0] 
    //
    //      attempts to arrange mappings
    //      as view the cube from the 6 different
    //      directions to yield consistent face winding  
    //      
    //            (x,y) -> (u,v)
    //            (y,z) -> (u,v)
    //            (x,z) -> (u,v)
    // 

    unsigned s = uv.s() ; 
    unsigned iu = uv.u() ; 
    unsigned iv = uv.v() ;
    unsigned nu = uv.nu() ; 
    unsigned nv = uv.nv();

    assert(s < par_nsurf());
    float fu = float(iu)/float(nu) ;
    float fv = float(iv)/float(nv) ;

    switch(s)
    {
        case 0:{    // -Z
                  p.x = glm::mix( bb.min.x, bb.max.x, 1 - fu ) ;
                  p.y = glm::mix( bb.min.y, bb.max.y, fv ) ;
                  p.z = bb.min.z ;
               } 
               ; break ;
        case 1:{   // +Z
                  p.x = glm::mix( bb.min.x, bb.max.x, fu ) ;
                  p.y = glm::mix( bb.min.y, bb.max.y, fv ) ;
                  p.z = bb.max.z ;
               }
               ; break ;


        case 2:{   // -X
                  p.x = bb.min.x ;
                  p.y = glm::mix( bb.min.y, bb.max.y, 1 - fu ) ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               } 
               ; break ;
        case 3:{   // +X
                  p.x = bb.max.x ;
                  p.y = glm::mix( bb.min.y, bb.max.y, fu ) ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               }
               ; break ;
 


        case 4:{  // -Y
                  p.x = glm::mix( bb.min.x, bb.max.x, fu ) ;
                  p.y = bb.min.y ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               } 
               ; break ;
        case 5:{  // +Y
                  p.x = glm::mix( bb.min.x, bb.max.x, 1 - fu ) ;
                  p.y = bb.max.y ;
                  p.z = glm::mix( bb.min.z, bb.max.z, fv ) ;
               }
               ; break ;
 

 
    }

/*
    std::cout << "nbox::par_pos"
              << " uv " << glm::to_string(uv) 
              << " p " << glm::to_string(p)
              << std::endl 
               ; 
*/

    return p ; 
}















void nbox::adjustToFit(const nbbox& bb, float scale)
{
    nquad qce ; 
    qce.f = bb.center_extent() ; 
    qce.f.w *= scale ; 

    init_box( *this, qce );
}





nbbox nbox::bbox() const
{
    nbbox bb ;
    if(is_box3)
    {
        bb.min = make_nvec3( -param.f.x/2.f, -param.f.y/2.f, -param.f.z/2.f );
        bb.max = make_nvec3(  param.f.x/2.f,  param.f.y/2.f,  param.f.z/2.f );
    }
    else
    {
        float s  = param.f.w ; 
        bb.min = make_nvec3( center.x - s, center.y - s, center.z - s );
        bb.max = make_nvec3( center.x + s, center.y + s, center.z + s );
    }

    bb.side = bb.max - bb.min ; 

    return gtransform ? bb.transform(gtransform->t) : bb ; 

    // bbox transforms need TR not IR*IT as they apply directly to geometry 
    // unlike transforming the SDF point or ray tracing ray which needs the inverse irit 
}

glm::vec3 nbox::gseedcenter()
{
    return gtransform == NULL ? center : glm::vec3( gtransform->t * glm::vec4(center, 1.f ) ) ;
}

void nbox::pdump(const char* msg, int verbosity )
{
    std::cout 
              << std::setw(10) << msg 
              << " label " << ( label ? label : "no-label" )
              << " center " << center 
              << " side " << param.f.w 
              << " gseedcenter " << gseedcenter()
              << " gtransform? " << !!gtransform
              << " is_box3 " << is_box3
              << std::endl ; 

    if(verbosity > 1 && gtransform) std::cout << *gtransform << std::endl ;
}





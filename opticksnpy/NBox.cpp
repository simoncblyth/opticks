#include "PLOG.hh"

#include "GLMFormat.hpp"
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

float nbox::operator()(float x_, float y_, float z_) const 
{
    glm::vec3 pos(x_,y_,z_);
    return sdf_(pos, gtransform);
} 

float nbox::sdf_model(const glm::vec3& pos) const { return sdf_(pos, NULL) ; }
float nbox::sdf_local(const glm::vec3& pos) const { return sdf_(pos, transform) ; }
float nbox::sdf_global(const glm::vec3& pos) const { return sdf_(pos, gtransform) ; }

float nbox::sdf_(const glm::vec3& pos, NNodeFrameType fty) const 
{
    float sd = 0.f ; 
    switch(fty)
    {
       case FRAME_MODEL  : sd = sdf_(pos, NULL)      ; break ; 
       case FRAME_LOCAL  : sd = sdf_(pos, transform) ; break ; 
       case FRAME_GLOBAL : sd = sdf_(pos, gtransform) ; break ; 
    }
    return sd ; 
}

float nbox::sdf_(const glm::vec3& pos, const nmat4triple* triple) const 
{
    glm::vec4 q(pos,1.0); 

    if(triple) q = triple->v * q ;  // NB using inverse transform on the query point 

    glm::vec3 c = center();

    glm::vec3 p = glm::vec3(q) - c ;  // coordinates in frame with origin at box center 

    glm::vec3 a = glm::abs(p) ;

    glm::vec3 h = halfside();

    glm::vec3 d = a - h ; 

    float sd = glm::compMax(d) ;

    return complement ? -sd : sd ; 
}

float nbox::sdf1(float x_, float y_, float z_) const 
{
    return (*this)(x_,y_,z_);
}

float nbox::sdf2(float x_, float y_, float z_) const 
{
    glm::vec4 p(x_,y_,z_,1.0); 

    if(gtransform) p = gtransform->v * p ;  

    glm::vec3 bmx = bmax() ; 

    // abs query point folds 3d space into +ve octant
    // subtract bmx places box max at origin in this folded space  

    glm::vec3 d = glm::abs(glm::vec3(p)) - bmx  ;

    float dmaxcomp = glm::compMax(d);

    glm::vec3 dmax = glm::max( d, glm::vec3(0.f) );

    float d_inside = fminf(dmaxcomp, 0.f);
    float d_outside = glm::length( dmax );

    return d_inside + d_outside ;       

   // see tests/NBoxTest.cc   sdf2 and sdf1 match despite code appearances
}

/*
http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
*/



glm::vec3 nbox::bmax() const 
{
    glm::vec3 h = halfside(); 
    glm::vec3 c = center();    
    if(is_box3) assert( c.x == 0 && c.y == 0 && c.z == 0 );
    return c + h ; 
}

glm::vec3 nbox::bmin() const 
{
    glm::vec3 h = halfside(); 
    glm::vec3 c = center();    
    if(is_box3) assert( c.x == 0 && c.y == 0 && c.z == 0 );
    return c - h ; 
}

nbbox nbox::bbox_model() const
{
    glm::vec3 bmi = bmin() ; 
    glm::vec3 bmx = bmax() ; 

    nbbox bb = make_bbox() ;
    bb.min = make_nvec3( bmi.x, bmi.y, bmi.z  );
    bb.max = make_nvec3( bmx.x, bmx.y, bmx.z  );
    bb.invert = complement ; 

    return bb ; 
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



glm::vec3 nbox::par_pos_model( const nuv& uv) const 
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

    //nbbox bb = bbox() ;  // NB bbox() has gtransform->t is already applied
    nbbox bb = bbox_model() ;  // bbox_model() has no transforms applied

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
    std::cout << "nbox::par_pos_model"
              << " uv " << glm::to_string(uv) 
              << " p " << glm::to_string(p)
              << std::endl 
               ; 
*/
    
    return p ; 
}






void nbox::nudge(unsigned s, float delta)
{
/*
How to nudge to avoid a coincidence ? 

* want to grow the coincident face by some delta eg ~2*epsilon 

* but CSG_BOX3 is positioned at 0,0,0 in model frame with 3 dimensions 
  ... so need to grow eg +Z face to avoid coincidence, but that 
  would require a compensating translation transform ???

* CSG_BOX is model frame placed but is symmetric, 
  hmm could non-uniformly scale OR rather just not support as CSG_BOX is only used in
  testing not real geometry.

* Does it matter that grow both +Z and -Z by a few epsilon, when only +Z is coincident ?  
  Currently envisage such nudges being applied only to subtracted 
  sub-objects, eg B in (A - B) or B in A*!B 

  YES, it does matter, consider (A - B) cutting a groove...
  growing out into empty space in +Z for the subtracted box B
  is needed to avoid the edge speckles and does not change 
  the geometry, (because there is no interesction between this growth and A)

  Whereas growing B in -Z would deepen the groove. 

             _______        +Z
       +-----+ . . +-----+   ^
       |     |  B  |     |   |  
       |  A  +-----+     |
       |                 |
       +-----------------+


* DO I NEED A SEPARATE nudge_transform BECAUSE HOW MUCH TO DELTA TRANSLATE 
  TO PREVENT MOVEMENT OF THE FIXED FACE DEPENDS ON CURRENT TRANSFORM
  (IF ITS NOT JUST A TRANSLATION WHICH COMMUTES)

  * this is mixed frame thinking... just think within the single primitive
    frame, just want to grow the box in one direction... if you compensate to 
    fix the face in the prim frame, it will be fixed in the composite one 

  * gtransforms are normally created for all primitives 
    when NCSG::import_r recursion gets down to them based on the
    heirarchy of transforms collected from the ancestors of the primitive in 
    the tree ... so that means need to change transform and then update gtransforms



*/
    assert( s < par_nsurf());
    assert( is_box3 && type == CSG_BOX3 && "nbox::nudge only implemented for CSG_BOX3 not CSG_BOX " );

    if(verbosity > 0)
    {
        std::cout << "nbox::nudge START" 
                  << " s " << s 
                  << " delta " << delta
                  << " param.f " << param.f.desc()
                  << " verbosity " << verbosity
                  << std::endl ;
    }

    glm::vec3 tlate(0,0,0) ; 

    // -Z :  grows in +-Z,  tlate in -Z keeping +Z in same position
    // +Z :  grows in +-Z,  tlate in +Z keeping -Z in same positiom 

    switch(s)
    {
        case 0:{ param.f.z += delta ; tlate.z = -delta/2.f ; } ; break ; // -Z
        case 1:{ param.f.z += delta ; tlate.z =  delta/2.f ; } ; break ; // +Z

        case 2:{ param.f.x += delta ; tlate.x = -delta/2.f ; } ; break ; // -X
        case 3:{ param.f.x += delta ; tlate.x =  delta/2.f ; } ; break ; // +X

        case 4:{ param.f.y += delta ; tlate.y = -delta/2.f ; } ; break ; // -Y 
        case 5:{ param.f.y += delta ; tlate.y =  delta/2.f ; } ; break ; // +Y 
    }

    // HMM MAYBE PREFERABLE TO NON-UNIFORMLY SCALE TO ACHIEVE A DELTA
    // RATHER THAN ACTUALLY CHANGING GEOMETRY, AS THAT CAN BE DONE TO ANYTHING ???

    if(verbosity > 0)
    std::cout << "nbox::nudge DONE " 
              << " s " << s 
              << " delta " << delta
              << " param.f " << param.f.desc()
              << " tlate " << glm::to_string(tlate)
              << std::endl ;


    if(transform == NULL) transform = nmat4triple::make_identity() ; 

    glm::mat4 compensate_ = nglmext::make_translate(tlate); 
    nmat4triple compensate(compensate_);

    bool reverse = true ;   // <-- think of the nudge as an origin frame inside the leaf 
    std::vector<const nmat4triple*> triples ; 
    triples.push_back(&compensate); 
    triples.push_back(transform); 
    const nmat4triple* compensated = nmat4triple::product(triples, reverse );

    // cannot use make_translated as need the translation first ...
    // transform = transform->make_translated(tlate, reverse, "nbox::nudge" ); 

    if(verbosity > 0)
    std::cout << "nbox::nudge"
              << std::endl 
              << gpresent("compensate_", compensate_ )
              << std::endl 
              << " changing primitive transform "
              << std::endl
              << gpresent("transform->t", transform->t) 
              << std::endl
              << " with "
              << std::endl
              << gpresent("compensate.t", compensate.t)
              << std::endl
              << " -> "
              << gpresent("compensated->t", compensated->t)
              << std::endl
              ;

    transform = compensated ; 
    gtransform = global_transform();  // product of transforms from heirarchy


}




void nbox::adjustToFit(const nbbox& bb, float scale)
{
    nquad qce ; 
    qce.f = bb.center_extent() ; 
    qce.f.w *= scale ; 

    init_box( *this, qce );
}


nbbox nbox::bbox_(NNodeFrameType fr) const 
{
    nbbox bb = bbox_model();
    nbbox tbb(bb);

    if(fr == FRAME_LOCAL && transform)
    {
        tbb = bb.make_transformed(transform->t);
    }
    else if(fr == FRAME_GLOBAL && gtransform)
    {
        tbb = bb.make_transformed(gtransform->t);
    }
    return tbb ; 
}

nbbox nbox::bbox_(const nmat4triple* triple) const
{
    nbbox bb = bbox_model();
    return triple ? bb.make_transformed(triple->t) : bb ; 
    // bbox transforms need TR not IR*IT as they apply directly to geometry 
    // unlike transforming the SDF point or ray tracing ray which needs the inverse irit 
}


nbbox nbox::bbox()        const { return bbox_(FRAME_GLOBAL) ; } 
nbbox nbox::bbox_global() const { return bbox_(FRAME_GLOBAL) ; } 
nbbox nbox::bbox_local()  const { return bbox_(FRAME_LOCAL) ; } 




glm::vec3 nbox::gseedcenter() const 
{
    return gtransform == NULL ? center() : glm::vec3( gtransform->t * glm::vec4(center(), 1.f ) ) ;
}

void nbox::pdump(const char* msg) const 
{
    std::cout 
              << std::setw(10) << msg 
              << " nbox::pdump "
              << " label " << ( label ? label : "-" )
              << " center " << center() 
              << " param.f " << param.f.desc() 
              << " side " << param.f.w 
              << " gseedcenter " << gseedcenter()
              << " transform? " << !!transform
              << " gtransform? " << !!gtransform
              << " is_box3 " << is_box3
              << std::endl ; 


    if(verbosity > 1 && transform) 
         std::cout 
              << std::endl
              << gpresent("tr->t", transform->t) 
              << std::endl
              ;

    if(verbosity > 1 && gtransform) 
         std::cout 
              << std::endl
              << gpresent("gtr->t", gtransform->t) 
              << std::endl
              ;

    if(verbosity > 2)
    {
        dumpSurfacePointsAll("nbox::pdump", FRAME_MODEL );
        dumpSurfacePointsAll("nbox::pdump", FRAME_LOCAL );
        dumpSurfacePointsAll("nbox::pdump", FRAME_GLOBAL );
    }
}






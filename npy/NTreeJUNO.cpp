#include <sstream>
#include <iomanip>
#include "NNode.hpp"
#include "NTreeJUNO.hpp"
#include "NSolid.hpp"

#include "NGLMExt.hpp"
#include "GLMPrint.hpp"
#include "NTorus.hpp"
#include "NSphere.hpp"
#include "NZSphere.hpp"
#include "NCone.hpp"

#include "Ellipse.hpp"

#include "PLOG.hh"


const NTreeJUNO::VI NTreeJUNO::LVS = { 18,-18, 19,-19, 20,-20, 21,-21  } ; 

nnode* NTreeJUNO::create(int lv)   // static
{
    //assert( std::find( LVS.begin(), LVS.end(), lv ) != LVS.end() );  

    nnode* a = NSolid::create( lv < 0 ? -lv : lv ) ; 
    nnode* b = lv < 0 ? NTreeJUNO::Rationalize( a ) : a ; 
    return b ;  
}


nnode* NTreeJUNO::Rationalize(nnode* a)   // static
{
    NTreeJUNO tj(a) ; 
    tj.rationalize(); 
    return tj.root ; 
}


NTreeJUNO::NTreeJUNO(nnode* root_ ) 
    :
    root(root_),
    cone(replacement_cone())
{
}

ncone* NTreeJUNO::replacement_cone() const 
{
    // ana/shape.py ana/x018.py 

    ntorus* t = (ntorus*)root->find_one(CSG_TORUS) ; 
    assert(t); 
    float R = t->rmajor();    
    float r = t->rminor();    

    const nmat4triple* tt = t->global_transform(); 
    assert(tt); 
    print( tt->t, "torus->t" ); 
    glm::vec3 tla = nglmext::pluck_translation( tt->t ) ; 
    print( tla, " tla " ); 
    float z0 = tla.z  ; //  torus z-plane in ellipsoid frame 

    glm::dvec2 torus_rhs(R, z0) ;   
    LOG(info) << " torus_rhs " << glm::to_string( torus_rhs) ; 
    

    LOG(info) << "torus R " << R << " r " << r ; 

    const nnode* sp = root->find_one(CSG_SPHERE, CSG_ZSPHERE) ; 

    glm::vec3 axes ; 
    glm::vec2 zcut ; 
    nnode::reconstruct_ellipsoid( sp , axes, zcut );  
    assert( axes.x == axes.y && "rotational symmetry about z is assumed" );  
    LOG(info) << " ellipsoid axes " << glm::to_string( axes ) << " zcut " << glm::to_string(zcut)  ; 

    ellipse e(axes.x, axes.z) ; 
    glm::dvec2 ca = e.closest_approach_to_point( torus_rhs );
    
    LOG(info) << " ca " << glm::to_string(ca) ; 

    float r2 = ca.x ;  
    float z2 = ca.y ;
    float r1 = R - r ;
    float z1 = z0 ;   

    float mz = (z1 + z2)/2.f ;  // mid-z cone coordinate (ellipsoid frame)
    float hz = (z2 - z1)/2.f ;  // cone half height 

    ncone* cn = make_cone( r1,-hz,r2,hz) ;
    cn->transform = nmat4triple::make_translate(0,0,mz); 

    return cn ; 
}


/**
NTreeJUNO::rationalize
-----------------------

18
    root gets replaced with ellipsoid
19
    root->left becomes root
19,20,21
    tubs-torus replaced with cone    

**/


void NTreeJUNO::rationalize()   // cf ana/shape.py ana/x018.py 
{
     LOG(info); 

     const nnode* e = root->find_one(CSG_ZSPHERE, CSG_SPHERE);
     assert(e); 
     nnode* ellipsoid = const_cast<nnode*>(e) ; 


     const nnode* t = root->find_one(CSG_TORUS);
     assert(t); 
  
     nnode* ss = t->parent ;                     // subtraction solid parent of torus
     assert( ss && ss->type == CSG_DIFFERENCE ); 

     nnode* us = ss->parent ;                    // union solid grandparent of torus
     assert( us && us->type == CSG_UNION ); 

     assert( us->left && us->left == e && us->right == ss && ss->right == t && us->right->right == t );

     assert( root->label ); 

     bool is_x018 = strcmp(root->label, NSolid::x018_label) == 0 ; 
     bool is_x019 = strcmp(root->label, NSolid::x019_label) == 0 ; 
     bool is_x020 = strcmp(root->label, NSolid::x020_label) == 0 ; 
     bool is_x021 = strcmp(root->label, NSolid::x021_label) == 0 ; 

     LOG(info)
           << " label " << root->label 
           << " is_x018 " << is_x018 
           << " is_x019 " << is_x019 
           << " is_x020 " << is_x020 
           << " is_x021 " << is_x021
           ; 

     if( is_x018 )  // cathode vacuum cap
     {
         assert( root->type == CSG_INTERSECTION ); 
         ellipsoid->parent = NULL ;

         root = ellipsoid ;              // copy/steal ?
     }
     else if( is_x019 )  // vacuum remainder
     {
         assert( root->type == CSG_DIFFERENCE ); 
         nnode* left = root->left ; 
         assert( left->type == CSG_UNION ); 
         left->parent = NULL ;

         root = left ; 
     }

     if( is_x019 || is_x020 || is_x021 )
     {
          cone->parent = us ; 
          us->right = cone ;           
     }
}






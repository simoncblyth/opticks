#pragma once
/**
csg_intersect_leaf.h : distance_leaf and intersect_leaf functions
===================================================================

Thus header needs to be included before csg_intersect_node.h which needs to be included before csg_intersect_tree.h 

distance_leaf_sphere 
intersect_leaf_sphere
    CSG_SPHERE, robust_quadratic_roots

distance_leaf_zsphere 
intersect_leaf_zsphere
    CSG_ZSPHERE, robust_quadratic_roots 

distance_leaf_convexpolyhedron
intersect_leaf_convexpolyhedron
    CSG_CONVEXPOLYHEDRON, plane intersections

MISSING : distance_leaf_cone
intersect_leaf_cone
    CSG_CONE, newcone with robust_quadratic_roots, oldcone without

MISSING : distance_leaf_hyperboloid
intersect_leaf_hyperboloid
    CSG_HYPERBOLOID, robust_quadratic_roots 

distance_leaf_box3
intersect_leaf_box3
    CSG_BOX3, plane intersections

distance_leaf_plane
intersect_leaf_plane
    CSG_PLANE

MISSING : distance_leaf_phicut 
intersect_leaf_phicut
    CSG_PHICUT 

distance_leaf_slab
intersect_leaf_slab
    CSG_SLAB

distance_leaf_cylinder
intersect_leaf_cylinder
    CSG_CYLINDER, robust_quadratic_roots_disqualifying 

MISSING : distance_leaf_infcylinder
intersect_leaf_infcylinder
    CSG_INFCYLINDER, robust_quadratic_roots

MISSING : distance_leaf_disc
intersect_leaf_disc
    CSG_DISC, disc still using the pseudo-general flop-heavy approach similar to oldcylinder
  
    * TODO: adopt less-flops approach like newcylinder
    * (NOT URGENT AS disc NOT CURRENTLY VERY RELEVANT IN ACTIVE GEOMETRIES) 

distance_leaf
intersect_leaf

Bringing over functions from  ~/opticks/optixrap/cu/csg_intersect_primitive.h

**/

#include "csg_intersect_leaf_head.h"
#include "OpticksCSG.h"
#include "squad.h"

#include "CSGNode.h"
#include "CSGPrim.h"

#include "csg_robust_quadratic_roots.h"
#include "csg_classify.h"

#if !defined(PRODUCTION) && defined(DEBUG_RECORD)
#include <csignal>
#endif

#if !defined(PRODUCTION) && defined(DEBUG_CYLINDER)
#include "CSGDebug_Cylinder.hh"
#endif

#include "csg_intersect_leaf_sphere.h"
#include "csg_intersect_leaf_zsphere.h"
#include "csg_intersect_leaf_cylinder.h"
#include "csg_intersect_leaf_box3.h"
#include "csg_intersect_leaf_newcone.h"
#include "csg_intersect_leaf_convexpolyhedron.h"
#include "csg_intersect_leaf_hyperboloid.h"

#if !defined(PRODUCTION) && defined(CSG_EXTRA)
#include "csg_intersect_leaf_plane.h"
#include "csg_intersect_leaf_slab.h"
#include "csg_intersect_leaf_phicut.h"
#include "csg_intersect_leaf_thetacut.h"
#include "csg_intersect_leaf_oldcone.h"
#include "csg_intersect_leaf_oldcylinder.h"
#include "csg_intersect_leaf_infcylinder.h"
#include "csg_intersect_leaf_disc.h"
#endif


/**
distance_leaf
---------------

For hints on how to implement distance functions for more primitives:

* https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
* env-;sdf-

**/

LEAF_FUNC
float distance_leaf( const float3& global_position, const CSGNode* node, const float4* plan, const qat4* itra )
{
    const unsigned typecode = node->typecode() ;  
    const unsigned gtransformIdx = node->gtransformIdx() ; 
    const bool complement = node->is_complement();

    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None

    float3 local_position  = q ? q->right_multiply(global_position,  1.f) : global_position ;  
    float distance = 0.f ;  

    switch(typecode)
    {
        case CSG_SPHERE:           distance = distance_leaf_sphere(            local_position, node->q0 )           ; break ; 
        case CSG_ZSPHERE:          distance = distance_leaf_zsphere(           local_position, node->q0, node->q1 ) ; break ; 
        case CSG_CYLINDER:         distance = distance_leaf_cylinder(          local_position, node->q0, node->q1 ) ; break ;
        case CSG_BOX3:             distance = distance_leaf_box3(              local_position, node->q0 )           ; break ;
        case CSG_CONE:             distance = 0.f                                                                   ; break ; 
        case CSG_CONVEXPOLYHEDRON: distance = distance_leaf_convexpolyhedron(  local_position, node, plan )         ; break ;
        case CSG_HYPERBOLOID:      distance = 0.f                                                                   ; break ; 
#if !defined(PRODUCTION) && defined(CSG_EXTRA)
        case CSG_PLANE:            distance = distance_leaf_plane(             local_position, node->q0 )           ; break ;
        case CSG_SLAB:             distance = distance_leaf_slab(              local_position, node->q0, node->q1 ) ; break ;
        case CSG_OLDCYLINDER:      distance = distance_leaf_cylinder(          local_position, node->q0, node->q1 ) ; break ;
        case CSG_PHICUT:           distance = distance_leaf_phicut(            local_position, node->q0 )           ; break ;
#endif
    }

    const float sd = complement ? -distance : distance  ; 
#if !defined(PRODUCTION) && defined(DEBUG)
    printf("//distance_leaf typecode %d name %s complement %d sd %10.4f \n", typecode, CSG::Name(typecode), complement, sd  ); 
#endif
    return sd ; 
}


/**
intersect_leaf : must be purely single node 
----------------------------------------------

Notice that only the inverse CSG transforms are needed on the GPU as these are used to 
transform the ray_origin and ray_direction into the local origin and direction in the 
local frame of the node.   

**/

LEAF_FUNC
bool intersect_leaf( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction )
{
    const unsigned typecode = node->typecode() ;  
    const unsigned gtransformIdx = node->gtransformIdx() ; 
    const bool complement = node->is_complement();

    const qat4* q = gtransformIdx > 0 ? itra + gtransformIdx - 1 : nullptr ;  // gtransformIdx is 1-based, 0 meaning None

    float3 origin    = q ? q->right_multiply(ray_origin,    1.f) : ray_origin ;  
    float3 direction = q ? q->right_multiply(ray_direction, 0.f) : ray_direction ;   

#if !defined(PRODUCTION) && defined(DEBUG_RECORD)
    printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
#endif

#if !defined(PRODUCTION) && defined(DEBUG)
    //printf("//[intersect_leaf typecode %d name %s gtransformIdx %d \n", typecode, CSG::Name(typecode), gtransformIdx ); 
    //printf("//intersect_leaf ray_origin (%10.4f,%10.4f,%10.4f) \n",  ray_origin.x, ray_origin.y, ray_origin.z ); 
    //printf("//intersect_leaf ray_direction (%10.4f,%10.4f,%10.4f) \n",  ray_direction.x, ray_direction.y, ray_direction.z ); 
    /*
    if(q) 
    {
        printf("//intersect_leaf q.q0.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q0.f.x,q->q0.f.y,q->q0.f.z,q->q0.f.w  ); 
        printf("//intersect_leaf q.q1.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q1.f.x,q->q1.f.y,q->q1.f.z,q->q1.f.w  ); 
        printf("//intersect_leaf q.q2.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q2.f.x,q->q2.f.y,q->q2.f.z,q->q2.f.w  ); 
        printf("//intersect_leaf q.q3.f (%10.4f,%10.4f,%10.4f,%10.4f)  \n", q->q3.f.x,q->q3.f.y,q->q3.f.z,q->q3.f.w  ); 
        printf("//intersect_leaf origin (%10.4f,%10.4f,%10.4f) \n",  origin.x, origin.y, origin.z ); 
        printf("//intersect_leaf direction (%10.4f,%10.4f,%10.4f) \n",  direction.x, direction.y, direction.z ); 
    }
    */
#endif

    bool valid_isect = false ; 
    switch(typecode)
    {
        case CSG_SPHERE:           valid_isect = intersect_leaf_sphere(           isect, node->q0,               t_min, origin, direction ) ; break ; 
        case CSG_ZSPHERE:          valid_isect = intersect_leaf_zsphere(          isect, node->q0, node->q1,     t_min, origin, direction ) ; break ; 
        case CSG_CYLINDER:         valid_isect = intersect_leaf_cylinder(         isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_BOX3:             valid_isect = intersect_leaf_box3(             isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_CONE:             valid_isect = intersect_leaf_newcone(          isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_CONVEXPOLYHEDRON: valid_isect = intersect_leaf_convexpolyhedron( isect, node, plan,             t_min, origin, direction ) ; break ;
        case CSG_HYPERBOLOID:      valid_isect = intersect_leaf_hyperboloid(      isect, node->q0,               t_min, origin, direction ) ; break ;
#if !defined(PRODUCTION) && defined(CSG_EXTRA)
        case CSG_PLANE:            valid_isect = intersect_leaf_plane(            isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_SLAB:             valid_isect = intersect_leaf_slab(             isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_OLDCYLINDER:      valid_isect = intersect_leaf_oldcylinder(      isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_PHICUT:           valid_isect = intersect_leaf_phicut(           isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_THETACUT:         valid_isect = intersect_leaf_thetacut(         isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_OLDCONE:          valid_isect = intersect_leaf_oldcone(          isect, node->q0,               t_min, origin, direction ) ; break ;
        case CSG_INFCYLINDER:      valid_isect = intersect_leaf_infcylinder(      isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
        case CSG_DISC:             valid_isect = intersect_leaf_disc(             isect, node->q0, node->q1,     t_min, origin, direction ) ; break ;
#endif
    }
    // NB: changing typecode->imp mapping is a handy way to use old imp with current geometry 


    if(valid_isect)
    {
        if(q) q->left_multiply_inplace( isect, 0.f ) ;  
        // normals transform differently : with inverse-transform-transposed 
        // so left_multiply the normal by the inverse-transform rather than the right_multiply 
        // done above to get the inverse transformed origin and direction
        //const unsigned boundary = node->boundary();  ???

        if(complement)  // flip normal for complement 
        {
            isect.x = -isect.x ;
            isect.y = -isect.y ;
            isect.z = -isect.z ;
        }
    }   
    else
    {
         // even for miss need to signal the complement with a -0.f in isect.x
         if(complement) isect.x = -isect.x ;  
         // note that isect.y is also flipped for unbounded exit : for consumption by intersect_tree
    }


#if !defined(PRODUCTION) && defined(DEBUG_RECORD)
    printf("//]intersect_leaf typecode %d name %s valid_isect %d isect (%10.4f %10.4f %10.4f %10.4f)   \n", typecode, CSG::Name(typecode), valid_isect, isect.x, isect.y, isect.z, isect.w); 
#endif

    return valid_isect ; 
}


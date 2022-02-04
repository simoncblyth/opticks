#pragma once
/**
csg_intersect_node.h
=======================

This header must be included after csg_intersect_leaf.h and before csg_intersect_tree.h

distance_node_contiguous
intersect_node_contiguous
     CSG_CONTIGUOUS shapes are multiunions that the user guarantees to have the topology of a ball  

TODO: distance_node_discontiguous, intersect_node_discontiguous 
     CSG_DISCONTIGUOUS shapes are multiunions that the user guarantees to have no overlap between nodes, 
     (perhaps CSG_DISJOINT or CSG_SEPERATED would be better names)
     for example a menagerie of different primitives arranged such that there are all entirely disconnected
     from each other 


intersect_node
distance_node


**/


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define INTERSECT_FUNC __forceinline__ __device__
#else
#    define INTERSECT_FUNC inline
#endif



/**
distance_node_contiguous
----------------------------

HMM : I think this same implementation should work for distance_node_discontiguous too 

**/


INTERSECT_FUNC
float distance_node_contiguous( const float3& pos, const CSGNode* node, const float4* plan, const qat4* itra )
{
    const unsigned num_sub = node->subNum() ; 
    float sd = RT_DEFAULT_MAX ; 
    for(unsigned isub=0 ; isub < num_sub ; isub++)
    {
         const CSGNode* sub_node = node+1u+isub ; 
         float sub_sd = distance_leaf( pos, sub_node, plan, itra ); 
         sd = fminf( sd, sub_sd ); 
    }
    return sd ; 
}


/**
intersect_node_contiguous
----------------------------

Example of a contiguous shape composed of constituent boxes::


                                           +---------------+
                                           |               |   
                                           |               |   
                                           |               |   
                             +-------------|...............|-----------+
                             |             .               .           |   
                             |             .               .           |   
                             |             .               .           |   
                             |             .................           |   
                             |                                         |   
                             |                                         |   
                      +------........                            .......--------+
                      |      .      .                            .     .        |   
                      |      .      .                            .  0 -1 - - - [2]  
                      |      .      .                            .     .        |   
           0 - - - - [1]- - -2      .                     0 - - -1 - - 2 - - - [3]  
                      |      .      .                            E     X        |   
                      |      .      .                            .     .        |  
                      +------........                            .......--------+
                             |                                         |   
                             |                                         |   
                             |                                   0 - - 1   
                             |                                         |   
                             |             .................           |   
                             |             .               .           |   
                     0 - - - 1 - - - - - - 2               .           |   
                             E             E               .           |   
                             +-------------.................-----------+
                                           |               |   
                                           |               |   
                                           |               |   
                                           +---------------+



* can tell are outside all constituents (and hence the compound) 
  when *ALL* first intersects on constituents are state ENTER 
  [this assumes non-complemented constituents]
  (this is because EXIT is shielded by the corresponding ENTER) 

  * -> compound intersect is the closest ENTER

* when *ANY* EXIT states are obtained from first intersects this means 
  are inside the CONTIGUOUS compound

  * this true in general for CONTIGUOUS compounds as are considering intersects 
    with all constituents and constituents do not "shield" each other : so will 
    always manage to have a first intersect that EXIT when start inside any of them 

  * to find the compound intersect are forced to find all the EXITS 
    for all the consituents that found ENTERs for  

**/


INTERSECT_FUNC
bool intersect_node_contiguous( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    float sd = distance_node_contiguous( ray_origin, node, plan, itra ); 
    bool inside_or_surface = sd <= 0.f ;
 
    const unsigned num_sub = node->subNum() ; 

    float4 isect_nearest_enter = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ; 
    float4 isect_farthest_exit = make_float4( 0.f, 0.f, 0.f, t_min ) ; 

    float4 sub_isect_0 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    
    float4 sub_isect_1 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    

    unsigned enter_count = 0 ; 
    unsigned exit_count = 0 ; 
    float propagate_epsilon = 0.0001f ; 

    for(unsigned isub=0 ; isub < num_sub ; isub++)
    {
        const CSGNode* sub_node = node+1u+isub ; 
        if(intersect_leaf( sub_isect_0, sub_node, plan, itra, t_min, ray_origin, ray_direction ))
        {
             IntersectionState_t sub_state_0 = CSG_CLASSIFY( sub_isect_0, ray_direction, t_min ); 
             if( sub_state_0 == State_Exit)
             {
                 exit_count += 1 ; 
                 if( sub_isect_0.w > isect_farthest_exit.w ) isect_farthest_exit = sub_isect_0 ;  
             }
             else if( sub_state_0 == State_Enter)
             {
                 enter_count += 1 ; 
                 if( sub_isect_0.w < isect_nearest_enter.w ) isect_nearest_enter = sub_isect_0 ;  

                 if(inside_or_surface)  
                 // when inside_or_surface need to find EXITs for all the ENTERs, when ouside just need nearest ENTER
                 { 
                     float tminAdvanced = sub_isect_0.w + propagate_epsilon ; 
                     if(intersect_leaf( sub_isect_1, sub_node, plan, itra, tminAdvanced , ray_origin, ray_direction ))
                     {
                          IntersectionState_t sub_state_1 = CSG_CLASSIFY( sub_isect_1, ray_direction, tminAdvanced ); 
                          if( sub_state_1 == State_Exit ) 
                          {
                              exit_count += 1 ; 
                              if( sub_isect_1.w > isect_farthest_exit.w ) isect_farthest_exit = sub_isect_1 ;  
                          } 
                     }
                 }
             }
        }
    }

    bool valid_intersect = false ; 
    if( inside_or_surface )
    {
        valid_intersect = exit_count > 0 && isect_farthest_exit.w > t_min ; 
        if(valid_intersect) isect = isect_farthest_exit ;  
    } 
    else
    {
        valid_intersect = enter_count > 0 && isect_nearest_enter.w > t_min ; 
        if(valid_intersect) isect = isect_nearest_enter ;  
    }
    return valid_intersect ; 
}


/**
intersect_node_discontiguous
-----------------------------

The guarantee that all sub-nodes do not overlap other sub-nodes
makes the implementation straightforward and hence fast:

* origin outside: closest ENTER
* origin inside: closest EXIT 

Providing more types of multiunion allows the user to communicate 
more precisely hence a simpler less general algorithm more suited 
to the situation can be applied. 

**/








/**
intersect_node : some node are compound eg CSG_CONTIGUOUS, but mostly leaf nodes
----------------------------------------------------------------------------------

Three level layout tree-node-leaf is needed to avoid intersect_node recursion which OptiX disallows. 

**/


INTERSECT_FUNC
bool intersect_node( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction )
{
    const unsigned typecode = node->typecode() ;  
    bool valid_intersect ; 
    switch(typecode)
    {
       case CSG_CONTIGUOUS:  valid_intersect = intersect_node_contiguous(isect, node, plan, itra, t_min, ray_origin, ray_direction )  ; break ; 
                   default:  valid_intersect = intersect_leaf(           isect, node, plan, itra, t_min, ray_origin, ray_direction )  ; break ; 
    }
    return valid_intersect ; 
}

/**
distance_node
----------------

**/

INTERSECT_FUNC
float distance_node( const float3& global_position, const CSGNode* node, const float4* plan, const qat4* itra )
{
    const unsigned typecode = node->typecode() ;  
    float distance ; 
    switch(typecode)
    {
        case CSG_CONTIGUOUS: distance = distance_node_contiguous( global_position, node, plan, itra )  ; break ; 
                    default: distance = distance_leaf(            global_position, node, plan, itra )  ; break ; 
    }
    return distance ; 
}


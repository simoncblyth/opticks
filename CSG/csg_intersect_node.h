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
             E : ENTER                     |               |   
             X : EXIT                      |               |   
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
                      E      E      .                            E     X        |   
                      |      .      .                            .     .        |  
                      +------........                            .......--------+
                             |                                         |   
                             |                                         |   
                             |                                   0 - - 1   
                             |                                         |   
                             |             .................           |   
                             |             .               .           |   
                     0 - - -[1]- - - - - - 2               .           |   
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

  * this is true in general for CONTIGUOUS compounds as are considering intersects 
    with all constituents and constituents do not "shield" each other : so will 
    always manage to have a first intersect that EXITs when start inside any of them 

  * to find the compound intersect are forced to find all the EXITS 
    for all the consituents that found ENTERs for  


ALG ISSUE : DOES NOT HONOUR t_min CUTTING : PRESUMABLY DUE TO MIXING DISTANCE AND INTERSECT ?

* fix by checking insideness of : ray_origin + t_min*ray_direction  ?
* TODO: check get expected t_min behaviour, that is: cutaway sphere in perspective projection around origin
  (correct t_min behaviour is a requirement to participate in CSG trees)


PROBLEMS WITH USING DISTANCE FUNCTION 

Using distance_node_contiguous feels "impure" as it mixes volume-centric with surface-centric approaches
plus it comes with some issues:

1. kinda duplicitous : as will be running intersect_leaf on all sub_node anyhow
2. using a second way of getting the same info (with another float cut) feels likely to cause edge issues 
3. distance functions are not yet implemented for all leaf 
  
ADVANTAGE OF USING DISTANCE FUNCTION 

Knowing inside_or_surface ahead of time allows:

1. single loop over leaves (BUT there is a hidden loop inside distance_node_contiguous)
2. avoids storing ENTER/EXIT states and distances for isect  
 
   * expect this to be is a big GPU performance advantage (more storage less in flight)
   * especially advantageous as this shape is targetting CSG boolean abuse solids with large numbers of leaves 

THOUGHTS ON ROOT CAUSE

Need to know all isect are ENTER before can decide that do not need to find the EXITs.
Put another way : if any EXIT encountered, must promote all ENTER into EXIT, 
and doing this requires have the ENTER distance. 
If the EXIT is obtained immediately after the ENTER that avoids having to store the ENTER distance. 
  
How important is the only ENTER optimization is unclear : would need to measure for specific geometry.
BUT : really want to avoid storing all isect if at all possible. 

FORGO ALL ENTER "OPTIMIZATION"

* no need to know inside_or_surface 
* just one loop and promote all ENTER to EXIT would also avoid the need to store all isect 


TODO: implement in several different ways and test performance for a variety of shapes

**/


INTERSECT_FUNC
bool intersect_node_contiguous( float4& isect, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    float sd = distance_node_contiguous( ray_origin + t_min*ray_direction, node, plan, itra ); 
    bool inside_or_surface = sd <= 0.f ;
 
    const unsigned num_sub = node->subNum() ; 

    float4 isect_nearest_enter = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ; 
    float4 isect_farthest_exit = make_float4( 0.f, 0.f, 0.f, t_min ) ; 

    float4 sub_isect_0 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    
    float4 sub_isect_1 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    
    // HMM: are both these sub_isect needed ? seems not : there is no comparison between them, 
    // HMM: same with sub_state_0, sub_state_1  

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

                 // when inside_or_surface need to find EXITs for all the ENTERs, when outside just need nearest ENTER
                 if(inside_or_surface)  
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
                          // TODO: debug check do not get another ENTER
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


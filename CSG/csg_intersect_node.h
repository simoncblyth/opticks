#pragma once
/**
csg_intersect_node.h
=======================

Providing more types of multi-union or multi-intersection allows the user to communicate 
more precisely hence a simpler less general algorithm more suited 
to the specific geometry can be applied. 



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
distance_node_list
----------------------------

1. get the number of subs from *node* which should be the list header node
2. loop over those subs

**/


INTERSECT_FUNC
float distance_node_list( unsigned typecode, const float3& pos, const CSGNode* node, const CSGNode* root, const float4* plan, const qat4* itra )
{
    const unsigned num_sub = node->subNum() ; 
    const unsigned offset_sub = node->subOffset(); 

    float sd = typecode == CSG_OVERLAP ? -RT_DEFAULT_MAX : RT_DEFAULT_MAX ; 
    for(unsigned isub=0 ; isub < num_sub ; isub++)
    {
         //const CSGNode* sub_node = node+1u+isub ;  
         // TOFIX: the abobe is assuming the sub_node follow the node, which they do not for lists within trees
         const CSGNode* sub_node = root+offset_sub+isub ;
#ifdef DEBUG
         printf("//distance_node_list num_sub %d offset_sub %d isub %d sub_node.typecode %d sub_node.typecode.name %s\n", num_sub, offset_sub, isub, sub_node->typecode(), CSG::Name(sub_node->typecode())) ;  
         assert( sub_node->typecode() > CSG_LEAF ); 
#endif

 
         float sub_sd = distance_leaf( pos, sub_node, plan, itra ); 

         switch(typecode)
         {
            case CSG_CONTIGUOUS:    sd = fminf( sd, sub_sd );   break ; 
            case CSG_DISCONTIGUOUS: sd = fminf( sd, sub_sd );   break ; 
            case CSG_OVERLAP:       sd = fmaxf( sd, sub_sd );   break ; 
         } 
#ifdef DEBUG
        printf("//distance_node_list isub %d sub_sd %10.4f sd %10.4f \n", isub, sub_sd, sd );  
#endif
    }
    return sd ; 
}


/**
intersect_node_contiguous : union of shapes which combine to make single fully connected compound shape 
------------------------------------------------------------------------------------------------------------

* shapes that filfil the requirements for using CSG_CONTIGUOUS (multi-union) can avoid tree overheads, 
  and thus benefit from simpler and faster intersection 
  

Example of a contiguous union shape composed of constituent boxes::


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
and doing this requires the ENTER distance. 
If the EXIT is obtained immediately after the ENTER that avoids having to store the ENTER distance. 
  
How important is the only ENTER optimization is unclear : would need to measure for specific geometry.
BUT : really want to avoid storing all isect if at all possible. 

FORGO ALL ENTER "OPTIMIZATION"

* no need to know inside_or_surface 
* just one loop and promote all ENTER to EXIT would also avoid the need to store all isect 


TODO: implement in several different ways and test performance for a variety of shapes

**/


INTERSECT_FUNC
bool intersect_node_contiguous( float4& isect, const CSGNode* node, const CSGNode* root, 
       const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
#ifdef DEBUG
     printf("//intersect_node_contiguous \n"); 
#endif

    float sd = distance_node_list( CSG_CONTIGUOUS, ray_origin + t_min*ray_direction, node, root, plan, itra ); 
    bool inside_or_surface = sd <= 0.f ;
    const unsigned num_sub = node->subNum() ; 
    const unsigned offset_sub = node->subOffset() ; 

#ifdef DEBUG
     printf("//intersect_node_contiguous sd %10.4f inside_or_surface %d num_sub %d offset_sub %d \n", sd, inside_or_surface, num_sub, offset_sub ); 
#endif

    float4 nearest_enter = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ; 
    float4 farthest_exit = make_float4( 0.f, 0.f, 0.f, t_min ) ; 

    float4 sub_isect_0 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    
    float4 sub_isect_1 = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    
    // HMM: are both these sub_isect needed ? seems not : there is no comparison between them, 
    // HMM: same with sub_state_0, sub_state_1  

    unsigned enter_count = 0 ; 
    unsigned exit_count = 0 ; 
    float propagate_epsilon = 0.0001f ; 

    for(unsigned isub=0 ; isub < num_sub ; isub++)
    {
        const CSGNode* sub_node = root+offset_sub+isub ; 
 
        if(intersect_leaf( sub_isect_0, sub_node, plan, itra, t_min, ray_origin, ray_direction ))
        {
             IntersectionState_t sub_state_0 = CSG_CLASSIFY( sub_isect_0, ray_direction, t_min ); 
             if( sub_state_0 == State_Exit)
             {
                 exit_count += 1 ; 
                 if( sub_isect_0.w > farthest_exit.w ) farthest_exit = sub_isect_0 ;  
             }
             else if( sub_state_0 == State_Enter)
             {
                 enter_count += 1 ; 
                 if( sub_isect_0.w < nearest_enter.w ) nearest_enter = sub_isect_0 ;  

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
                              if( sub_isect_1.w > farthest_exit.w ) farthest_exit = sub_isect_1 ;  
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
        valid_intersect = exit_count > 0 && farthest_exit.w > t_min ; 
        if(valid_intersect) isect = farthest_exit ;  
    } 
    else
    {
        valid_intersect = enter_count > 0 && nearest_enter.w > t_min ; 
        if(valid_intersect) isect = nearest_enter ;  
    }

#ifdef DEBUG
     printf("//intersect_node_contiguous valid_intersect %d  (%10.4f %10.4f %10.4f %10.4f) \n", valid_intersect, isect.x, isect.y, isect.z, isect.w ); 
#endif

    return valid_intersect ; 
}



/**
intersect_node_discontiguous : union of disjoint leaves with absolutely no overlapping
----------------------------------------------------------------------------------------

The guarantee that all sub-nodes do not overlap other sub-nodes
makes the implementation straightforward and hence fast:

* closest ENTER or EXIT 
**/


INTERSECT_FUNC
bool intersect_node_discontiguous( float4& isect, const CSGNode* node, const CSGNode* root, 
     const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    const unsigned num_sub = node->subNum() ; 
    const unsigned offset_sub = node->subOffset() ; 

    float4 closest = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ; 
    float4 sub_isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    

    for(unsigned isub=0 ; isub < num_sub ; isub++)
    {
        const CSGNode* sub_node = root+offset_sub+isub ; 
        if(intersect_leaf( sub_isect, sub_node, plan, itra, t_min, ray_origin, ray_direction ))
        {
            if( sub_isect.w < closest.w ) closest = sub_isect ;  
        }
    }

    bool valid_isect = closest.w < RT_DEFAULT_MAX ; 
    if(valid_isect) 
    {
        isect = closest ;  
    }

#ifdef DEBUG
    printf("//intersect_node_discontiguous num_sub %d  closest.w %10.4f \n", 
       num_sub, closest.w ); 
#endif

    return valid_isect ; 
}
 






/**
intersect_node_overlap : intersection of leaves
----------------------------------------------------

* HMM: could general sphere be implemented using CSG_OVERLAP multi-intersection 
  of inner and outer sphere, phi planes and theta cones ?

* shapes that can be described entirely with intersection can avoid tree overheads 



Imagine the CSG_OVERLAP of 3 shapes A,B,C the resulting ABC shape is the portion that is inside all of them.

Algorithm loops over constituent sub nodes classifying intersects as ENTER/EXIT/MISS whilst 
updating : nearest_exit, farthest_enter. 

* from outside the compound ABC shape the desired intersect is the deepest enter ENTER
  which you can find by shooting A,B,C and 


                                                       1X,2M -> MISS

                                        +----------------1X-------------+
                                        |               /             C |
                                        |              /                |
                                        |             0                 |
                                        |                               |
                              +---------------------------------+       |  
                              |         |                     B |       |  
                              |         |                       |       |  
                   +----------------------------+         0- - -1 - - - 2     2X,1M -> MISS
                   | A        |         | . ABC |               X       X  
                   |          |         | . . . |               |       |  
                   |          |         | . . . |               |       |  
          0 - - - -1- - - - - 2- - - - [3]- - - 4               |       |  
                   E          E         E . . . X               |       |  
                   |          |         | . . . |               |       |  
                   |          |         | . . . |               |       |  
                   |          |         | .0- -[1]- - - - - - - 2 - - - 3 - - 
                   |          |         | . . . X               X       X  
                   |          |         | . . . |               |       |  
                   |          |         +-------------------------------+  
                   |          |                 |               |          
                   |          |                 |               |          
             0 - - 1 - - - - -2 - - - - - - - - 3 - - - - - - - 4          
                   E          E   (2E,1M)-> M   X               X   (2X,1M) -> M
                   |          |                 |               |          
                   |          +---------------------------------+          
                   |                            |                          
                   |                            |                                
                   +----------------------------+                          
             



The below arrangement of constituents are not permissable as there is no common overlap
giving such a shape to a CSG_OVERLAP may abort and will give incorrect or no intersects.      


             +------+  
         +---|------|-----+
         |   |      |     |
         |   |      |     |
         |   +------+     | 
         |           +----|-----+
         +-----------|----+     |
                     +----------+

      +----+      +-----+      +-----+
      |    |      |     |      |     |
   - -E - - - - - E- - - - - - E- - - - - - 
      |    |      |     |      |     |
      +----+      +-----+      +-----+



If eed to compare the farthest enter with the nearest enter 
if nearest enter is closer than farthest exit then its a HIT ?



Thinking of ROTH diagrams for intersection

                           
                       |--------------|
                       
                            |------------------|

                                  |-----------------|

           Enters:     |    |     | 

           Exits:                     |        |    |


                  farthest enter  |   |
                                  |   |    nearest exit

For an overlap need::

                      farthest_enter < nearest_exit  


For disjoint::


                          |-----------|     

                                           |------------|
  
            Enters:       |                |

            Exits:                    |                 |              
                         
  
                         nearest exit |    | farthest enter 

No overlap as::

                         nearest_exit   < farthest_enter 




**/


INTERSECT_FUNC
bool intersect_node_overlap( float4& isect, const CSGNode* node, const CSGNode* root, 
        const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    const unsigned num_sub = node->subNum() ; 
    const unsigned offset_sub = node->subOffset() ; 

    float4 farthest_enter = make_float4( 0.f, 0.f, 0.f, t_min ) ; 
    float4 nearest_exit  = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ; 
    float4 sub_isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    

    float propagate_epsilon = 0.0001f ; 
    unsigned enter_count = 0 ; 
    unsigned exit_count = 0 ; 
    IntersectionState_t sub_state = State_Miss ; 

    for(unsigned isub=0 ; isub < num_sub ; isub++)
    {
        const CSGNode* sub_node = root+offset_sub+isub ; 
        if(intersect_leaf( sub_isect, sub_node, plan, itra, t_min, ray_origin, ray_direction ))
        {
            sub_state = CSG_CLASSIFY( sub_isect, ray_direction, t_min ); 
            if(sub_state == State_Enter)
            {
                enter_count += 1 ;  
                if( sub_isect.w > farthest_enter.w ) farthest_enter = sub_isect ;  

                float tminAdvanced = sub_isect.w + propagate_epsilon ; 
                if(intersect_leaf( sub_isect, sub_node, plan, itra, tminAdvanced , ray_origin, ray_direction ))
                {
                    sub_state = CSG_CLASSIFY( sub_isect, ray_direction, tminAdvanced ); 
                    if( sub_state == State_Exit ) 
                    {
                        exit_count += 1 ;  
                        if( sub_isect.w < nearest_exit.w ) nearest_exit = sub_isect ;  
                    } 
                }
            }
            else if(sub_state == State_Exit)
            {
                exit_count += 1 ; 
                if( sub_isect.w < nearest_exit.w ) nearest_exit = sub_isect ;  
            }
        } 
    }


    // if no Enter encountered farthest_enter.w stays t_min
    // if no Exit encountered nearest_exit.w  stays RT_DEFAULT_MAX 

    bool valid_isect = false ;  
    bool overlap_all = farthest_enter.w < nearest_exit.w && max(enter_count, exit_count) == num_sub ; 
    if(overlap_all)  
    {
        if( farthest_enter.w > t_min && farthest_enter.w  < RT_DEFAULT_MAX )
        {
            valid_isect = true ; 
            isect = farthest_enter ; 
        }
        else if( nearest_exit.w > t_min && nearest_exit.w < RT_DEFAULT_MAX )
        {
            valid_isect = true ; 
            isect = nearest_exit ; 
        }
    }

#ifdef DEBUG
    printf("//intersect_node_overlap num_sub %d  enter_count %d exit_count %d overlap_all %d valid_isect %d farthest_enter.w %10.4f nearest_exit.w %10.4f \n", 
       num_sub, enter_count, exit_count, overlap_all, valid_isect, farthest_enter.w, nearest_exit.w ); 
#endif

    return valid_isect ; 
}
 








/**
intersect_node : some node are compound eg CSG_CONTIGUOUS, but mostly leaf nodes
----------------------------------------------------------------------------------

Three level layout tree-node-leaf is needed to avoid intersect_node recursion which OptiX disallows. 

**/


INTERSECT_FUNC
bool intersect_node( float4& isect, const CSGNode* node, const CSGNode* root, 
       const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin , const float3& ray_direction )
{
    const unsigned typecode = node->typecode() ;  

#ifdef DEBUG
    printf("//intersect_node typecode %d name %s \n", typecode, CSG::Name(typecode) ); 
#endif

    bool valid_intersect ; 
    switch(typecode)
    {
       case CSG_CONTIGUOUS:     valid_intersect = intersect_node_contiguous(   isect, node, root, plan, itra, t_min, ray_origin, ray_direction )  ; break ; 
       case CSG_OVERLAP:        valid_intersect = intersect_node_overlap(      isect, node, root, plan, itra, t_min, ray_origin, ray_direction )  ; break ; 
       case CSG_DISCONTIGUOUS:  valid_intersect = intersect_node_discontiguous(isect, node, root, plan, itra, t_min, ray_origin, ray_direction )  ; break ; 
                   default:     valid_intersect = intersect_leaf(              isect, node,       plan, itra, t_min, ray_origin, ray_direction )  ; break ; 
    }

    return valid_intersect ; 
}

/**
distance_node
----------------

**/

INTERSECT_FUNC
float distance_node( const float3& global_position, const CSGNode* node, const CSGNode* root, const float4* plan, const qat4* itra )
{
    const unsigned typecode = node->typecode() ;  
    float distance ; 
    switch(typecode)
    {
        case CSG_CONTIGUOUS: 
        case CSG_OVERLAP:          distance = distance_node_list( typecode,  global_position, node, root, plan, itra )  ; break ; 
        case CSG_DISCONTIGUOUS:    distance = distance_node_list( typecode,  global_position, node, root, plan, itra )  ; break ; 
        default:                   distance = distance_leaf(                 global_position, node, plan, itra )  ; break ; 
    }
    return distance ; 
}


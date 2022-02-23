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
         assert( sub_node->typecode() >= CSG_LEAF ); 
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

* fixied by checking insideness of : ray_origin + t_min*ray_direction  ?
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
   * from future : note that cannot avoid storing t_enter to properly get furthest exit that is not disjoint 
     making the case to eliminate the distance function

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


ISSUE WHEN USED IN TREE

Checking this within a CSG tree boolean combination reveals that it is an 
over simplification to just return the furthest exit as it misses intermediate exits
that prevents the CSG tree intersection from working.

Enters and exits that are within the compound must not raise an intersect 
but when a enter or exit transitions in or out of the compound an intersect must 
be returned even when there is a subsequent one further along. 

How to do that without lots of distance calls ?
Perhaps a depth counter can distinguish between internal 
and external borders.



         +----------+ 
         |        B |
         |          |
    +----|----------|-----+ 
    | A  |          |     |
    |    |          |     |
    | 0 E1         X2    X3         In this simple case [3] is the last exit, but that is not generally the case 
    |   +1          0    -1         for more involved shapes. 
    |    |          |     |
    |    |          |     |         depth counter, starting at 0, add 1 at an Enter, subtract 1 at an Exit 
    |    |          |     |         when start inside must end -ve 
    |    |          |     |         
    |    |          |     |         
    |    |    0    X1    X2         
    |    |         -1    -2         
    |    |          |     |
    |    +----------+     |
    |                     |
    +---------------------+
    
    
But the alg works by looping over ALL constituents

* classifying Enter/Exit
* when get an Enter, also get the corresponding Exit

* there can be subsequnet Enters after an Exit, but that would have to be 
  handled by another call to the intersect with advanced tmin : because 
  only the next isect with the compound at t > tmin is returned


So in the above:

   A: Exit at t=3  (-1)
   B: Enter at t=1 (-1+1=0), Exit at t=2 (0-1 = -1) 



Now for A B C


         +----------+ 
         |        B |
         |          |
    +----|----------|-----+                      +-------------------+
    | A  |          |     |                      |                C  |  
    |    |          |     |                      |                   | 
    | 0 E1         X2   [X3]                    E4                  X5
    |   +1          0    -1                      0                  -1
    |    |          |     |                      |                   |                        
    |    |          |     |                      |                   |
    |    |          |     |                      |                   |
    |    |    0    X1   [X2]                    E3                  X4     
    |    |         -1    -2                     -1                  -2
    |    |          |     |                      |                   | 
    |    |          |     |                      |                   | 
    |    |          | 0 [X1]                    E2                  X3 
    |    |          |     -1                     0                   -1
    |    |          |     |                      |                   | 
    |    +----------+     |                      |                   |
    |                     |                      |                   |
    +---------------------+                      +-------------------+
 
                                                            

For rays starting in A or B need to disqualify C intersects because 
the C enters are at greater t than the A and B exits.

If C expanded to the left to join with A and B then that would not 
be the case and its exits would be allowed.  

Any of the A,B,C can be disjoint and need to detect and disqualify.

HMM: should any constituent with an Enter exceeding 

HMM: when are inside are in need of an exit, but if a constituent
gives you an enter it is under suspicion of needing disqualification.

HMM: need to distinguish first-exits from second-exits (obtained after an enter) 

The farthest first-exit is highly likely to be THE ONE, but 
not inevitably.  So on first loop collect the farthest first exit 

Perhaps can disqualify constituents with Enter that is 
further away that the farthest first-exit. 


What about a chain, perhaps repeated pushing 
of envelope prevents the two pass approach ?



                 +----------------+     +-------------------+ 
                 |B               |     |D                  |
                 |                |     |                   |
                 |                |     |                   |
            +----|----+      +----|-----|----+       +------|----------+       
            |A   |    |      |C   |     |    |       |E     |          |
            |    |    |      |    |     |    |       |      |          |
            | 0 E1   (X2)    E3  X4    E5   X6      E7     X8         X9
            |    |    |      |    |     |    |       |      |          |
            |    |    |      |    |     |    |       |      |          |
            |    |    |      |    |     |    |       |      |          |
            +----|----+      +----|-----|----+       +------|----------+   
                 |                |     |                   |
                 |                |     |                   |
                 |                |     |                   |
                 +----------------+     +-------------------+

                 E           E          E            E           
                      X           X          X              X          X       



   
1. *first pass* : find furthest first exit 

   A: X2 : this is the furthest first exit 
   B: E1
   C: E3
   D: E5
   E: E7

2. *second pass* : redo the undone that are all State_Enter, looping subs with enter less than furthest exit 
   
   * only E1 qualifies, looping that takes you to X4 






1. *zeroth pass* : hoping that are outside just find nearest enter and count first exits 

2. when no Exits are outside the compound which makes the intersect simply the closest enter, so return it  

3. *first pass* : collect enter distances, isub indices and get farthest_exit of the first exits 

   * first exits always qualify as potential intersects, it is only exits after an enter that may be disjoint 
     and require contiguity checking before can qualify as candidate intersect
    


**/


INTERSECT_FUNC
bool intersect_node_contiguous( float4& isect, const CSGNode* node, const CSGNode* root, 
       const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    const int num_sub = node->subNum() ; 
    const int offset_sub = node->subOffset() ; 
#ifdef DEBUG
     printf("//intersect_node_contiguous num_sub %d offset_sub %d \n", num_sub, offset_sub ); 
#endif

#ifdef DEBUG_DISTANCE
    float sd = distance_node_list( CSG_CONTIGUOUS, ray_origin + t_min*ray_direction, node, root, plan, itra ); 
    bool inside_or_surface = sd <= 0.f ;
    printf("//intersect_node_contiguous sd %10.4f inside_or_surface %d num_sub %d offset_sub %d \n", sd, inside_or_surface, num_sub, offset_sub ); 
#endif

    float4 nearest_enter = make_float4( 0.f, 0.f, 0.f, RT_DEFAULT_MAX ) ; 
    float4 farthest_exit = make_float4( 0.f, 0.f, 0.f, t_min ) ; 
    // TODO: with split zeroth and first passes do not need both these isect at the same time, so could combine/reuse

    float4 sub_isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ;    

    int exit_count = 0 ; 
    const float propagate_epsilon = 0.0001f ; 
    IntersectionState_t sub_state = State_Miss ;  

    // 1. *zeroth pass* : hoping that are outside just find nearest enter and count exits 

    for(int i=0 ; i < num_sub ; i++)
    {
        const CSGNode* sub_node = root+offset_sub+i ; 
        if(intersect_leaf( sub_isect, sub_node, plan, itra, t_min, ray_origin, ray_direction ))
        {
            sub_state = CSG_CLASSIFY( sub_isect, ray_direction, t_min ); 
#ifdef DEBUG
            printf("//intersect_node_contiguous i %d state %s t %10.4f \n", i, IntersectionState::Name(sub_state), sub_isect.w );   
#endif          
            if( sub_state == State_Enter)
            {
                if( sub_isect.w < nearest_enter.w ) nearest_enter = sub_isect ;  
            }
            else if( sub_state == State_Exit )
            {
                exit_count += 1 ; 
            }
        }
    } 

#ifdef DEBUG
    printf("//intersect_node_contiguous exit_count %d \n", exit_count); 
#endif

    // 2. when no Exits are outside the compound which makes the intersect simply the closest enter, so return it
    if(exit_count == 0) 
    {
        bool valid_intersect = nearest_enter.w > t_min && nearest_enter.w < RT_DEFAULT_MAX ; 
        if(valid_intersect) isect = nearest_enter ;
#ifdef DEBUG
        printf("//intersect_node_contiguous : outside early exit   %10.4f %10.4f %10.4f %10.4f \n", isect.x, isect.y, isect.z, isect.w );  
#endif
        return valid_intersect ;   
    }


    // now to work, there are Exits so are inside the compund and need to get outta here using some resources
    // hmm but i guess these static resources will also be in play even with the early exit ?
    // TODO: compare performance between combining and splitting the zeroth and first passes 


    int enter_count = 0 ; 
    float enter[8] ; 
    int   aux[8] ; 
    int   idx[8] ;   // HMM: could squeeze 16 nibbles into the 64 bits 

    
    
    // 3. *first pass* : collect enter distances, isub indices and get farthest_exit of the first exits 
    // 
    // * first exits always qualify as potential intersects, it is only exits after an enter that may be disjoint 
    //   and require contiguity checking before can qualify as candidate intersect
    // 

    for(int isub=0 ; isub < num_sub ; isub++)
    {
        const CSGNode* sub_node = root+offset_sub+isub ; 
        if(intersect_leaf( sub_isect, sub_node, plan, itra, t_min, ray_origin, ray_direction ))
        {
            sub_state = CSG_CLASSIFY( sub_isect, ray_direction, t_min ); 
            if( sub_state == State_Enter)
            {
                aux[enter_count] = isub ;   // record which isub this enter corresponds to 
                idx[enter_count] = enter_count ; 
                enter[enter_count] = sub_isect.w ;

                // TODO: very wasteful enter_count and isub usually very small values, 
                //       could pack them together perhaps ?


                // HMM: trying to do two levels of indirection at once doesnt work with the indirect sort, hence have to use the aux

#ifdef DEBUG
                printf("//intersect_node_contiguous isub %d enter_count %d idx[enter_count] %d enter[enter_count] %10.4f \n", isub, enter_count, idx[enter_count], enter[enter_count] ); 
#endif
                enter_count += 1 ; 
            }
            else if( sub_state == State_Exit )  
            {
                exit_count += 1 ; 
                if( sub_isect.w > farthest_exit.w ) farthest_exit = sub_isect ;  
            }
        }
    } 

#ifdef DEBUG
    if(enter_count > 8)
    { 
        // maybe could use integer template specialization to tailor the limit to each geometry combined 
        // with nvrtc runtime-compilation to allow resource use customization during geometry conversion
        printf("//intersect_node_contiguous enter_count %d exceeds limit of 8 \n", enter_count );  
    }
    assert( enter_count <= 8 ) ; 
#endif


    // insertionSortIndirectSentinel : 
    // 4. order the enter indices so that they would make enter ascend 
    // see SysRap/tests/sorting/insertionSortIndirect.sh to understand the sort 
    // ordering the idx (isub indices) to make the enter values ascend 

#ifdef DEBUG
    printf("//intersect_node_contiguous enter_count %d \n", enter_count ); 
    for(int i=0 ; i < enter_count ; i++) printf(" i %2d idx[i] %2d enter[i] %10.4f \n", i, idx[i], enter[i] ); 
#endif

    for (int i = 1; i < enter_count ; i++)
    {   
        int key = idx[i] ;  // hold idx[1] out of the pack    
        //int akey = aux[i] ; 
        int j = i - 1 ;
        
        // descending j below i whilst find out of order  
        while( j >= 0 && enter[idx[j]] > enter[key] )   
        {   
            idx[j+1] = idx[j] ;    // i=1,j=0,idx[1]=idx[0]   assuming not ascending
            //aux[j+1] = aux[j] ;  
            
            j = j - 1;
            
            // sliding values (actually the isub indices) that are greater than the key one upwards
            // no need to "swap" as are holding the key out of the pack
            // ready to place it into the slot opened by the slide up   
        }
        
        idx[j+1] = key ;       // i=1,j=-1, idx[0]=key
        //aux[j+1] = akey ; 

        // i=1, j->0, when enter[j] <= enter[i]  (already ascending) 
        // the while block doesnt run 
        //   => pointless idx[1] = idx[1]   
        // so puts the key back in the pack at the same place it came from  
    }

 
    // *2nd pass* : loop over enters in t order  
    // only find exits for Enters that qualify as contiguous (thus excluding disjoint subs)
    // where the qualification is based on the farthest_exit.w 
    //
    // I think that because the enters are t ordered there is no need for rerunning this
    // despite each turn of the loop potentially pushing the envelope of permissible enters
    //
    // suspect can break on first disqualification ? think line of disjoint boxes
    //

    for(int i=0 ; i < enter_count ; i++)
    {
        //int isub = aux[i];  // reference back from enter count index  *i* to sub-index *isub*
        int isub = aux[idx[i]];  // rather than shuffling aux, can just use it a fixed mapping from enter index to isub index 

        const CSGNode* sub_node = root+offset_sub+isub ; 
        float tminAdvanced = enter[i] + propagate_epsilon ; 

#ifdef DEBUG
        printf("//intersect_node_contiguous i/enter_count/isub %d/%d/%d tminAdvanced %10.4f farthest_exit.w %10.4f  \n", i, enter_count, isub, tminAdvanced, farthest_exit.w ); 
#endif

        if(tminAdvanced < farthest_exit.w) 
        {  
            if(intersect_leaf( sub_isect, sub_node, plan, itra, tminAdvanced , ray_origin, ray_direction ))
            {
                sub_state = CSG_CLASSIFY( sub_isect, ray_direction, tminAdvanced ); 
                if( sub_state == State_Exit ) 
                {
                    exit_count += 1 ; 
                    if( sub_isect.w > farthest_exit.w ) farthest_exit = sub_isect ;  
                }
#ifdef DEBUG
                assert( sub_state == State_Exit ); 
#endif          
            }
        }
    }

    bool valid_intersect = exit_count > 0 && farthest_exit.w > t_min ; 
    if(valid_intersect) isect = farthest_exit ;  
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


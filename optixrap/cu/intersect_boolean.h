#pragma once


static __device__
void intersect_boolean_only_first( const uint4& prim, const uint4& identity )
{
    unsigned a_partIdx = prim.x + 1 ;  

    float3 a_normal = make_float3(0.f,0.f,1.f);

    float tA_min = propagate_epsilon ;  
    float tA     = tA_min ;  

    IntersectionState_t a_state = intersect_part( a_partIdx , tA_min, a_normal, tA ) ;


    if(a_state != Miss)
    {
        if(rtPotentialIntersection(tA))
        {
            shading_normal = geometric_normal = a_normal;
            instanceIdentity = identity ;

#ifdef BOOLEAN_DEBUG
            instanceIdentity.x = dot(a_normal, ray.direction) < 0.f ? 1 : 2 ;
#endif
            rtReportIntersection(0);
        }
    }
}



static __device__
void intersect_csg( const uint4& prim, const uint4& identity )
{
    // see opticks/dev/csg/node.py:Node.postOrderSequence

    const unsigned long long postorder_sequence[4] = 
         { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;

    unsigned partOffset = prim.x ; 
    unsigned numParts   = prim.y ;
    unsigned primIdx_   = prim.z ; 

   // complete binary trees have node counts (numParts)
   //       1, 3, 7, 15, 31
   // (+1)  2, 4, 8, 16, 32   adding one gets a power of 2, so its a single bit 
   // ffs_  2, 3, 4,  5,  6 
   //       1, 2, 3,  4,  5
   //
   //  leftmost for subtree reiteration : multiply by 2 until exceed nodecount 
   //  (which could exclude leaves)

    unsigned height = __ffs( numParts + 1 ) - 2 ; 

    // the below use height-1 in order to fly above the leaves
    unsigned long long postorder = postorder_sequence[height-1] ; 
    unsigned numNodes = (0x1 << (1+height-1)) - 1 ;

    unsigned leftmost = 1 ; // start at root using 1-based levelorder indexing
    while(leftmost <= numNodes) leftmost = leftmost*2 ; 


    rtPrintf("intersect_csg partOffset %u numParts %u primIdx_ %u height %u postorder %llx leftmost %u  \n", partOffset, numParts, primIdx_, height, postorder, leftmost );

    unsigned beginIdx = postorder & 0xF ; 
    unsigned endIdx = 1 ; 


    unsigned i = 0 ; 
    unsigned nodeIdx = beginIdx ;
    while(nodeIdx >= endIdx)
    {
         unsigned leftIdx = partOffset + nodeIdx*2 ; 
         unsigned rightIdx = partOffset + nodeIdx*2 + 1; 

         // TODO:is_leaf, adopt OpticksCSG_t enum instead of OpticksShape_t mask, so can select off that 

         //rtPrintf("intersect_csg i:%d nodeIdx %d leftIdx %d rightIdx %d partOffset %d \n", i, nodeIdx, leftIdx, rightIdx, partOffset );

         i += 1 ;
         nodeIdx = (postorder & (0xFull << i*4 )) >> i*4 ; 
    }
}


static __device__
void intersect_boolean( const uint4& prim, const uint4& identity )
{
    // NB LIMITED TO SINGLE BOOLEAN OPERATION APPLIED TO TWO BASIS SOLIDS, ie triplet trees

    // primFlags only available for root of tree,
    // operate from partBuffer for other nodes

    unsigned partOffset = prim.x ; 
    unsigned primIdx_   = prim.z ; 

    unsigned n_partIdx = partOffset ;    
    unsigned a_partIdx = partOffset + 1 ;   // SIMPLIFYING TRIPLET ASSUMPTION
    unsigned b_partIdx = partOffset + 2 ;  

    quad q1 ; 
    q1.f = partBuffer[4*n_partIdx+1];
    OpticksShape_t operation = (OpticksShape_t)q1.u.w ;

    rtPrintf("intersect_boolean primIdx_:%u n:%u a:%u b:%u operation:%u \n", primIdx_, n_partIdx, a_partIdx, b_partIdx, operation );

    float3 a_normal = make_float3(0.f,0.f,1.f);
    float3 b_normal = make_float3(0.f,0.f,1.f);

    float tA_min = ray.tmin ; // formerly propagate_epsilon and before that 0.f
    float tB_min = ray.tmin ;
    float tA     = 0.f ;
    float tB     = 0.f ;

    int ctrl = CTRL_LOOP_A | CTRL_LOOP_B ; 

    IntersectionState_t a_state = Miss ; 
    IntersectionState_t b_state = Miss ; 

    int count(0) ;  
    while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && count < 4 )
    {
        count++ ; 

        a_state = (ctrl & CTRL_LOOP_A) ? intersect_part( a_partIdx , tA_min, a_normal, tA ) : a_state ;
        b_state = (ctrl & CTRL_LOOP_B) ? intersect_part( b_partIdx , tB_min, b_normal, tB ) : b_state ;

        int actions = boolean_actions( operation , a_state, b_state );
        int act = boolean_decision( actions, tA <= tB );
        ctrl = boolean_ctrl( act );

        if(     ctrl == CTRL_LOOP_A) tA_min = tA ; 
        else if(ctrl == CTRL_LOOP_B) tB_min = tB ; 
    } 


    // hmm below passing to OptiX should probably be done in caller ?
    if( ctrl & (CTRL_RETURN_A | CTRL_RETURN_B | CTRL_RETURN_FLIP_B  ))
    {
        if(rtPotentialIntersection( ctrl == CTRL_RETURN_A ? tA : tB))
        {
            shading_normal = geometric_normal = ctrl == CTRL_RETURN_A ? 
                                                                           a_normal
                                                                      :
                                                                          ( ctrl == CTRL_RETURN_FLIP_B ? -b_normal : b_normal )
                                                                      ;
            instanceIdentity = identity ;
            rtReportIntersection(0);
        }
    } 

}




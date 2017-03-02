#pragma once


static __device__
void intersect_boolean_only_first( const uint4& prim, const uint4& identity )
{
    unsigned a_partIdx = prim.x + 1 ;  


    float tA_min = propagate_epsilon ;  
    float4 tt = make_float4(0.f,0.f,1.f, tA_min);

    IntersectionState_t a_state = intersect_part( a_partIdx , tA_min, tt ) ;


    if(a_state != Miss)
    {
        if(rtPotentialIntersection(tt.w))
        {
            shading_normal.x = geometric_normal.x = tt.x ;
            shading_normal.y = geometric_normal.y = tt.y ;
            shading_normal.z = geometric_normal.z = tt.z ;

            instanceIdentity = identity ;
            //instanceIdentity.x = dot(a_normal, ray.direction) < 0.f ? 1 : 2 ;

            rtReportIntersection(0);
        }
    }
}


#define CSG_STACK_SIZE 4

#define POSTORDER(i) ((postorder & (0xFull << (i)*4 )) >> (i)*4 ) 


static __device__
void intersect_csg( const uint4& prim, const uint4& identity )
{
    // see opticks/dev/csg/node.py:Node.postOrderSequence
    // sequence of levelorder indices in postorder, which has tree meaning 
    const unsigned long long postorder_sequence[4] = { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;

    unsigned partOffset = prim.x ; 
    unsigned numParts   = prim.y ;
    unsigned primIdx_   = prim.z ; 

    unsigned height = __ffs(numParts + 1) - 2 ; // assumes perfect binary tree node count, for height h:  2^(h+1) - 1 

    unsigned long long postorder = postorder_sequence[height-1] ; // height-1 in order to fly above the leaves
    unsigned numInternalNodes = (0x1 << (1+height-1)) - 1 ;

    // the tranche indices pick ranges of the postorder sequence



    // 0-based indices into postorder sequence
    int postorder_begin = 0 ; 
    int postorder_end = numInternalNodes ; 

    // leftmost, root indices already in the postorder sequence 
    //rtPrintf("intersect_csg partOffset %u numParts %u numInternalNodes %u primIdx_ %u height %u postorder %llx \n", partOffset, numParts, numInternalNodes, primIdx_, height, postorder );

    float tmin = 0.f ;  

    // allocate stacks

    float4 _lhs[CSG_STACK_SIZE] ; 
    int lhs = -1 ; 

    float4 _rhs[CSG_STACK_SIZE] ; 
    int rhs = -1 ; 

    float4 _tmin ; 
    uint4  _tranche ; 
    int tranche = -1 ;

    float4 miss = make_float4(0.f,0.f,1.f,0.f);

    tranche++ ;  // push
    setByIndex(_tranche, tranche, ((postorder_end & 0xffff) << 16) | (postorder_begin & 0xffff) )  ;
    setByIndex(_tmin,    tranche,  tmin ); 

    while (tranche >= 0)
    {
         float   tmin = getByIndex(_tmin, tranche);
         unsigned tmp = getByIndex(_tranche, tranche );
         unsigned begin = tmp & 0xffff ;
         unsigned end   = tmp >> 16 ;
         tranche-- ;                // pop, -1 means empty stack

         for(unsigned i=begin ; i < end ; i++)
         {
             // XXidx are 1-based levelorder tree indices
             unsigned nodeIdx = (postorder & (0xFull << i*4 )) >> i*4 ;   
             unsigned leftIdx = nodeIdx*2 ; 
             unsigned rightIdx = nodeIdx*2 + 1; 

             unsigned lhsLeftMostIdx = leftIdx ;
             unsigned rhsLeftMostIdx = rightIdx ;

             while(lhsLeftMostIdx*2 < numInternalNodes) lhsLeftMostIdx *= 2 ;  
             while(rhsLeftMostIdx*2 < numInternalNodes) rhsLeftMostIdx *= 2 ;  



             bool bileaf = leftIdx > numInternalNodes ; 

             quad q1 ; 
             q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];
             OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;


             float4 left  = make_float4(0.f,0.f,1.f,0.f);
             float4 right = make_float4(0.f,0.f,1.f,0.f);

             float tA_min = ray.tmin ; // formerly propagate_epsilon and before that 0.f
             float tB_min = ray.tmin ;

             int ctrl = CTRL_LOOP_A | CTRL_LOOP_B ; 

             IntersectionState_t a_state = Miss ; 
             IntersectionState_t b_state = Miss ; 

             int count(0) ;  
             while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && count < 4 )
             {
                count++ ; 

                if(ctrl & CTRL_LOOP_A)
                {
                    if(bileaf)  
                    {
                         a_state = intersect_part( partOffset+leftIdx-1 , tA_min, left  ) ;
                    }
                    else                             // operation node
                    {
                         if(lhs >= 0)
                         {
                             left = _lhs[lhs] ;  
                             lhs-- ;          // pop
                         }
                         else
                         {
                             left = miss ; // ERROR popping from empty
                         } 
                    }
                }

                if(ctrl & CTRL_LOOP_B)
                {
                    if(bileaf)  // leaf node
                    {
                         b_state = intersect_part( partOffset+rightIdx-1 , tB_min, right  ) ;
                    }
                    else                             // operation node
                    {
                         if(rhs >= 0)
                         {
                             right = _rhs[rhs] ;  
                             rhs-- ;          // pop
                         }
                         else
                         {
                             right = miss ; // ERROR popping from empty
                         } 
                    }
                }
 

                int actions = boolean_actions( operation , a_state, b_state );
                int act = boolean_decision( actions, left.w <= right.w );
                ctrl = boolean_ctrl( act );

                if(ctrl == CTRL_LOOP_A) 
                {
                    tA_min = left.w  ;  // epsilon ? 

                    if(!bileaf)   // left is not leaf
                    {
                         rhs++ ;   // push other side, as just popped it while reiterating this side
                         _rhs[rhs] = right ;    

                         tranche++ ;  // push
                         setByIndex(_tranche, tranche, ((numInternalNodes & 0xffff) << 16) | (i & 0xffff) )  ;
                         setByIndex(_tmin,    tranche,  tmin );

                         tranche++ ;  // push
                         setByIndex(_tranche, tranche, ((left_leftmost & 0xffff) << 16) | ((i+1) & 0xffff) )  ;
                         setByIndex(_tmin,    tranche,  tA_min );

                    } 


                } 
                else if(ctrl == CTRL_LOOP_B) 
                {
                    tB_min = right.w ;   // epsilon ?
                }
 
             }  // end ctrl while      


             // defer partOffset-ing until usage, as thats a technical detail that looses the tree meaning of the nodeIdx

             rtPrintf("intersect_csg i:%d nodeIdx %d leftIdx %d rightIdx %d partOffset %d \n", i, nodeIdx, leftIdx, rightIdx, partOffset );
         }
    }
}


static __device__
void intersect_boolean_triplet( const uint4& prim, const uint4& identity )
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
    OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

    //rtPrintf("intersect_boolean primIdx_:%u n:%u a:%u b:%u operation:%u \n", primIdx_, n_partIdx, a_partIdx, b_partIdx, operation );

    float4 left  = make_float4(0.f,0.f,1.f,0.f);
    float4 right = make_float4(0.f,0.f,1.f,0.f);

    float tA_min = ray.tmin ; // formerly propagate_epsilon and before that 0.f
    float tB_min = ray.tmin ;

    int ctrl = CTRL_LOOP_A | CTRL_LOOP_B ; 

    IntersectionState_t a_state = Miss ; 
    IntersectionState_t b_state = Miss ; 

    int count(0) ;  
    while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && count < 4 )
    {
        count++ ; 

        a_state = (ctrl & CTRL_LOOP_A) ? intersect_part( a_partIdx , tA_min, left  ) : a_state ;
        b_state = (ctrl & CTRL_LOOP_B) ? intersect_part( b_partIdx , tB_min, right ) : b_state ;

        int actions = boolean_actions( operation , a_state, b_state );
        int act = boolean_decision( actions, left.w <= right.w );
        ctrl = boolean_ctrl( act );

        if(     ctrl == CTRL_LOOP_A) tA_min = left.w  ;  // no epsilon ? 
        else if(ctrl == CTRL_LOOP_B) tB_min = right.w ; 
    } 


    // hmm below passing to OptiX should probably be done in caller ?
    if( ctrl & (CTRL_RETURN_A | CTRL_RETURN_B | CTRL_RETURN_FLIP_B  ))
    {
        if(rtPotentialIntersection( ctrl == CTRL_RETURN_A ? left.w : right.w ))
        {
            shading_normal = geometric_normal = ctrl == CTRL_RETURN_A ? 
                                                                           make_float3(left.x, left.y, left.z)
                                                                      :
                                                                          ( ctrl == CTRL_RETURN_FLIP_B ? -make_float3(right.x, right.y, right.z) : make_float3(right.x, right.y, right.z) )
                                                                      ;
            instanceIdentity = identity ;
            rtReportIntersection(0);
        }
    } 

}




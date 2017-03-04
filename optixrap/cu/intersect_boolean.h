#pragma once


static __device__
void intersect_boolean_only_first( const uint4& prim, const uint4& identity )
{
    unsigned a_partIdx = prim.x + 1 ;  


    float tA_min = propagate_epsilon ;  
    float4 tt = make_float4(0.f,0.f,1.f, tA_min);

    //IntersectionState_t a_state = intersect_part( a_partIdx , tA_min, tt ) ;
    intersect_part( a_partIdx , tA_min, tt ) ;

    IntersectionState_t a_state = tt.w > tA_min ? 
                        ( (tt.x * ray.direction.x + tt.y * ray.direction.y + tt.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                              Miss
                              ; 

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
#define TRANCHE_STACK_SIZE 4

#define POSTORDER(i) ((postorder & (0xFull << (i)*4 )) >> (i)*4 ) 

/**

ideas

* copy boolean lookup from host side, to avoid all the case statements

* pack the three boolean lookup tables into 32bit uints and hold them in rtDeclareVariable(uint4,, ) = { 0xashvda, 0xasdgahsdb,  }

  * the tables are 3x3 (Enter,Exit,Miss)x(Enter,Exit,Miss) but can special case Miss, Miss to bring down to 8 elements
    that at 4 bits each can fit into 32-bits 
  * split the tables into ACloser and BCloser ones 
  
* use a Matrix<4,4> to holding rows with (miss,left,right,flip_right) so can refer to which by index
  rather than passing around float4

* lookup incorporating ACloser boolean, so cut out a lot of branches

* merge acts/act/ctrl logic

**/


static __device__
void intersect_csg( const uint4& prim, const uint4& identity )
{
    // see opticks/dev/csg/node.py:Node.postOrderSequence
    // sequence of 1-based levelorder indices in postorder, which has tree meaning 
    const unsigned long long postorder_sequence[4] = { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;

    int ierr = 0 ;  
    bool abort_ = false ; 

    unsigned partOffset = prim.x ; 
    unsigned numParts   = prim.y ;
    unsigned primIdx_   = prim.z ; 

    unsigned fullHeight = __ffs(numParts + 1) - 2 ;   // assumes perfect binary tree node count       2^(h+1) - 1 
    unsigned height = fullHeight - 1;                 // exclude leaves, triplet has height 0

    unsigned long long postorder = postorder_sequence[height] ; 
    unsigned numInternalNodes = (0x1 << (1+height)) - 1 ;

    float4 _lhs[CSG_STACK_SIZE] ; 
    int lhs = -1 ; 

    float4 _rhs[CSG_STACK_SIZE] ; 
    int rhs = -1 ; 

    float4 _tmin ;  // TRANCHE_STACK_SIZE is 4 
    uint4  _tranche ; 
    int tranche = -1 ;


    float4 miss   = make_float4(0.f,0.f,1.f,0.f);
    float4 result = make_float4(0.f,0.f,1.f,0.f) ; 

    tranche++ ;  // push 0-based postorder indices, initially selecting entire postorder sequence
    setByIndex(_tranche, tranche, ((numInternalNodes & 0xffff) << 16) | (0 & 0xffff) )  ; // postorder end, begin
    setByIndex(_tmin,    tranche,  ray.tmin );    // current tmin, (propagate_epsilon or 0.f not appropriate, eg when near-clipping)

    while (tranche >= 0)
    {
         float   tmin = getByIndex(_tmin, tranche);
         unsigned tmp = getByIndex(_tranche, tranche );
         unsigned begin = tmp & 0xffff ;
         unsigned end   = tmp >> 16 ;
         tranche-- ;                // pop, -1 means empty stack

         //rtPrintf("intersect_csg: begin %u end %u \n", begin, end );  // 0, 3

         for(unsigned i=begin ; i < end ; i++)
         {
             // XXidx are 1-based levelorder perfect tree indices
             unsigned nodeIdx = (postorder & (0xFull << i*4 )) >> i*4 ;   
             unsigned leftIdx = nodeIdx*2 ; 
             unsigned rightIdx = nodeIdx*2 + 1; 
             int depth = 32 - __clz(nodeIdx)-1 ;  
             unsigned subNodes = (0x1 << (1+height-depth)) - 1 ; // subtree nodes  
             unsigned halfNodes = (subNodes - 1)/2 ;             // nodes to left or right of subtree

             bool bileaf = leftIdx > numInternalNodes ; 


             quad q1 ; 
             q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];
             OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;


             float4 left  = make_float4(0.f,0.f,1.f,0.f);
             float4 right = make_float4(0.f,0.f,1.f,0.f);

             float tA_min = tmin ; 
             float tB_min = tmin ;

             int ctrl = CTRL_LOOP_A | CTRL_LOOP_B ; 
             bool reiterate = false ; 

             // TODO: try reording to reduce branchiness, as bileaf is constant for the loop
             // also could unroll, so the happy case of no looping goes more smoothly 


             int loop(-1) ;  
             while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && loop < 10 )
             {
                loop++ ; 

                if(ctrl & CTRL_LOOP_A)
                {
                    if(bileaf) // left leaf node 
                    {
                         intersect_part( partOffset+leftIdx-1 , tA_min, left  ) ;
                    }
                    else                             // operation node
                    {
                         if(lhs < 0)
                         {
                             ierr |= ERROR_LHS_POP_EMPTY ; 
                             left = miss ; 
                         } 
                         left = _lhs[lhs] ;  
                         lhs-- ;          // pop
                    }
                }       // CTRL_LOOP_A

                if(ctrl & CTRL_LOOP_B)
                {
                    if(bileaf)  // right leaf node
                    {
                         intersect_part( partOffset+rightIdx-1 , tB_min, right  ) ;
                    }
                    else        // operation node
                    {
                         if(rhs < 0)
                         {
                             ierr |= ERROR_RHS_POP_EMPTY ; 
                             abort_ = true ;
                             break ; 
                         } 
                         right = _rhs[rhs] ;  
                         rhs-- ;          // pop
                    }
                }       // CTRL_LOOP_B
 


                IntersectionState_t a_state = left.w > tA_min ? 
                        ( (left.x * ray.direction.x + left.y * ray.direction.y + left.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 

                IntersectionState_t b_state = right.w > tB_min ? 
                        ( (right.x * ray.direction.x + right.y * ray.direction.y + right.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 

                int actions = boolean_actions( operation , a_state, b_state );
                int act = boolean_decision( actions, left.w <= right.w );
                ctrl = boolean_ctrl( act );


                if(ctrl == CTRL_LOOP_A) 
                {
                    tA_min = left.w  ;  // epsilon ? 

                    if(!bileaf)   // left is not leaf
                    {
                         if(rhs+1 >= CSG_STACK_SIZE)
                         {
                             ierr |= ERROR_RHS_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         rhs++ ;   // push other side, as just popped it while reiterating this side
                         _rhs[rhs] = right ;    

                         if(tranche+2 >= TRANCHE_STACK_SIZE)
                         { 
                             ierr |= ERROR_LHS_TRANCHE_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         tranche++ ;  // push, from here on up : i -> numInternalNodes
                         setByIndex(_tranche, tranche, ((numInternalNodes & 0xffff) << 16) | (i & 0xffff) )  ;  
                         setByIndex(_tmin,    tranche,  tmin );

                         tranche++ ;  // push, left subtree  :  i - 2*halfNodes -> i - halfNodes
                         setByIndex(_tranche, tranche, ((i-halfNodes & 0xffff) << 16) | ((i-2*halfNodes) & 0xffff) )  ;
                         setByIndex(_tmin,    tranche,  tA_min );

                         reiterate = true ; 
                    } 
                } 
                else if(ctrl == CTRL_LOOP_B) 
                {
                    tB_min = right.w ;   // epsilon ?

                    if(!bileaf)   // left is not leaf
                    {
                         if(lhs+1 >= CSG_STACK_SIZE)
                         {
                             ierr |= ERROR_LHS_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         lhs++ ;   // push other side
                         _lhs[lhs] = left ;    


                         if(tranche+2 >= TRANCHE_STACK_SIZE)
                         { 
                             ierr |= ERROR_RHS_TRANCHE_OVERFLOW ; 
                             abort_ = true ;
                             break ; 
                         }

                         tranche++ ;  // push, from here on up : i -> numInternalNodes
                         setByIndex(_tranche, tranche, ((numInternalNodes & 0xffff) << 16) | (i & 0xffff) )  ;  
                         setByIndex(_tmin,    tranche,  tmin );

                         tranche++ ;  // push, right subtree :  i - halfNodes -> i
                         setByIndex(_tranche, tranche, ((i & 0xffff) << 16) | ((i-halfNodes) & 0xffff) )  ;
                         setByIndex(_tmin,    tranche,  tB_min );

                         reiterate = true ; 
                    } 
                }
                if(reiterate) break ;
             }  // end while : ctrl loop


             if(reiterate || abort_) break ;  
             // reiteration needs to get back to tranche loop for subtree traversal 
             // without "return"ing anything


             if( ctrl == CTRL_RETURN_MISS )
             {
                 result = miss ; 
             }
             else if(ctrl == CTRL_RETURN_A) 
             {
                 result = left ; 
             } 
             else if( ctrl == CTRL_RETURN_B )
             {
                 result = right ; 
             }
             else if( ctrl == CTRL_RETURN_FLIP_B )
             {
                 result.x = -right.x ; 
                 result.y = -right.y ; 
                 result.z = -right.z ; 
                 result.w =  right.w ; 
             }
             else
             {
                  ierr |= ERROR_BAD_CTRL ; 
             }   
         
             if(nodeIdx % 2 == 0) // even 1-based nodeIdx is left
             {
                 lhs++ ;   // push
                 _lhs[lhs] = result ;    
             }
             else
             {
                 rhs++ ;   // push
                 _rhs[rhs] = result ;    
             }

         }  // end for : node traversal within tranche
    }       // end while : tranche

    

    if(lhs != -1) ierr |= ERROR_LHS_END_NONEMPTY ;  
    if(rhs != 0)  ierr |= ERROR_RHS_END_EMPTY  ; 

    if(rhs == 0 && ierr == 0)
    {
         result = _rhs[rhs] ;  
         rhs-- ;  // pop
         if(rtPotentialIntersection( result.w ))
         {
              shading_normal = geometric_normal = make_float3(result.x, result.y, result.z) ;
              instanceIdentity = identity ;
              rtReportIntersection(0);
         }
    } 

    //rtPrintf("intersect_csg partOffset %u numParts %u numInternalNodes %u primIdx_ %u height %u postorder %llx ierr %x \n", partOffset, numParts, numInternalNodes, primIdx_, height, postorder, ierr );
    if(ierr != 0)
    rtPrintf("intersect_csg primIdx_ %u ierr %4x  (%10.3f %10.3f %10.3f %10.3f)   \n", primIdx_, ierr,  result.x, result.y, result.z, result.w  );

}   // intersect_csg


static __device__
void intersect_boolean_triplet( const uint4& prim, const uint4& identity )
{
    // NB LIMITED TO SINGLE BOOLEAN OPERATION APPLIED TO TWO BASIS SOLIDS, ie triplet trees

    // primFlags only available for root of tree,
    // operate from partBuffer for other nodes

    unsigned partOffset = prim.x ; 
    //unsigned primIdx_   = prim.z ; 

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


    int count(0) ;  
    while((ctrl & (CTRL_LOOP_A | CTRL_LOOP_B)) && count < 4 )
    {
        count++ ; 

        if(ctrl & CTRL_LOOP_A) intersect_part( a_partIdx , tA_min, left  ) ;
        if(ctrl & CTRL_LOOP_B) intersect_part( b_partIdx , tB_min, right ) ;

        IntersectionState_t a_state = left.w > tA_min ? 
                        ( (left.x * ray.direction.x + left.y * ray.direction.y + left.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 

        IntersectionState_t b_state = right.w > tB_min ? 
                        ( (right.x * ray.direction.x + right.y * ray.direction.y + right.z * ray.direction.z) < 0.f ? Enter : Exit ) 
                                  :
                                  Miss
                                  ; 

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




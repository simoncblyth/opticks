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


#define POSTORDER_NODE(postorder, i) (((postorder) & (0xFull << (i)*4 )) >> (i)*4 )
#define POSTORDER_SLICE(begin, end) (  (((end) & 0xffff) << 16) | ((begin) & 0xffff)  )
#define POSTORDER_BEGIN( tmp )  ( (tmp) & 0xffff )
#define POSTORDER_END( tmp )  ( (tmp) >> 16 )



#define CSG_PUSH(_stack, stack, ERR, val ) \
             { \
                 if((stack)+1 >= CSG_STACK_SIZE) \
                 {  \
                     ierr |= (ERR) ; \
                     abort_ = true ; \
                     break ;  \
                 }  \
                 (stack)++ ;   \
                 (_stack)[(stack)] = val ; \
             } \


#define CSG_POP(_stack, stack, ERR, ret ) \
             { \
                if((stack) < 0) \
                {  \
                    ierr |= (ERR)  ;\
                    abort_ = true ;  \
                    break ; \
                }  \
                (ret) = (_stack)[(stack)] ;  \
                (stack)-- ;    \
             } \
 

#define CSG_CLASSIFY( ise, dir, tmin )   ((ise).w > (tmin) ?  ( (ise).x*(dir).x + (ise).y*(dir).y + (ise).z*(dir).z < 0.f ? Enter : Exit ) : Miss )



#define TRANCHE_PUSH0( _stacku, _stackf, stack, valu, valf ) \
           { \
                (stack)++ ; \
                setByIndex( (_stacku), (stack), (valu) ) ; \
                setByIndex( (_stackf), (stack), (valf) ) ; \
           }  



#define TRANCHE_POP0( _stacku, _stackf, stack, valu, valf ) \
         (valf) = getByIndex((_stackf), (stack));  \
         (valu) = getByIndex((_stacku), (stack) ); \
         (stack)-- ; 
     

#define TRANCHE_PUSH( _stacku, _stackf, stack, ERR, valu, valf ) \
            { \
                if((stack)+1 >= TRANCHE_STACK_SIZE) \
                {   \
                     ierr |= (ERR) ; \
                     abort_ = true ; \
                     break ; \
                } \
                (stack)++ ; \
                setByIndex( (_stacku), (stack), (valu) ) ; \
                setByIndex( (_stackf), (stack), (valf) ) ; \
           }  



/**

TODO:

* find visual way to demonstrate where tranche mechanics actually working



**/


struct CSG 
{
   float4 data[CSG_STACK_SIZE] ; 
   int curr ;
};


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

    float4 _tmin ;  // TRANCHE_STACK_SIZE is 4 
    uint4  _tranche ; 
    int tranche = -1 ;

    enum { LHS, RHS };
    CSG csg[2] ; 
    CSG& lhs = csg[LHS] ; 
    CSG& rhs = csg[RHS] ; 
    lhs.curr = -1 ;  
    rhs.curr = -1 ; 

    enum { MISS, LEFT, RIGHT, RFLIP  } ;  // this order is tied to CTRL_ enum, needs rejig of lookup to change 
    float4 isect[4] ;
    isect[MISS]       =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[LEFT]       =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[RIGHT]      =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[RFLIP]      =  make_float4(0.f, 0.f, 1.f, 0.f);

    const float4& miss  = isect[MISS];
    float4& left  = isect[LEFT];
    float4& right = isect[RIGHT];
    float4& rflip = isect[RFLIP];

    TRANCHE_PUSH0( _tranche, _tmin, tranche, POSTORDER_SLICE(0, numInternalNodes), ray.tmin );

    while (tranche >= 0)
    {
         float   tmin ;
         unsigned tmp ;
         TRANCHE_POP0( _tranche, _tmin, tranche,  tmp, tmin );
         unsigned begin = POSTORDER_BEGIN( tmp );
         unsigned end   = POSTORDER_END( tmp );

         for(unsigned i=begin ; i < end ; i++)
         {
             // XXidx are 1-based levelorder perfect tree indices
             //unsigned nodeIdx = (postorder & (0xFull << i*4 )) >> i*4 ;   
             unsigned nodeIdx = POSTORDER_NODE(postorder, i) ;   
             unsigned leftIdx = nodeIdx*2 ; 
             unsigned rightIdx = nodeIdx*2 + 1; 
             int depth = 32 - __clz(nodeIdx)-1 ;  
             unsigned subNodes = (0x1 << (1+height-depth)) - 1 ; // subtree nodes  
             unsigned halfNodes = (subNodes - 1)/2 ;             // nodes to left or right of subtree
             bool bileaf = leftIdx > numInternalNodes ; 

             quad q1 ; 
             q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];      // (nodeIdx-1) as 1-based
             OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

             float tX_min[2] ; 
             float& tL_min = tX_min[LHS] ;
             float& tR_min = tX_min[RHS] ;
             tL_min = tmin ; 
             tR_min = tmin ;

             IntersectionState_t x_state[2] ; 

             // postorder traversal means that have always 
             // visited left and right subtrees before visiting a node

             if(bileaf) // op-left-right leaves
             {
                 left.w = 0.f ;   // reusing the same storage so clear ahead
                 right.w = 0.f ; 
                 intersect_part( partOffset+leftIdx-1 , tL_min, left  ) ;
                 intersect_part( partOffset+rightIdx-1 , tR_min, right  ) ;
             }
             else       //  op-op-op
             {
                 CSG_POP( lhs.data, lhs.curr, ERROR_LHS_POP_EMPTY, left );
                 CSG_POP( rhs.data, rhs.curr, ERROR_RHS_POP_EMPTY, right );
             }
 
             x_state[LHS] = CSG_CLASSIFY( left , ray.direction, tL_min ) ;
             x_state[RHS] = CSG_CLASSIFY( right, ray.direction, tR_min ) ;

             int ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], left.w <= right.w );
             int side = ctrl - CTRL_LOOP_A ;   // CTRL_LOOP_A,CTRL_LOOP_B -> LHS, RHS   looper side

             bool reiterate = false ; 
             int loop(-1) ;  
             while( side > -1 && loop < 10 )
             {
                 loop++ ; 
                 float4& _side = isect[side+LEFT] ; 

                 tX_min[side] = _side.w + propagate_epsilon ;   
            
                 if(bileaf)
                 {
                     // intersect lhs/rhs primitive again with tmin advanced
                      intersect_part( partOffset+leftIdx+side-1 , tX_min[side], _side  ) ;
                 }
                 else
                 {
                      // pop prior stacked result
                      CSG_POP( csg[side].data, csg[side].curr, ERROR_POP_EMPTY, _side );
                 }

                 // reclassification and boolean decision following advancement of one side 
                 x_state[side] = CSG_CLASSIFY( _side, ray.direction, tX_min[side] );
                 ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], left.w <= right.w );
                 side = ctrl - CTRL_LOOP_A ; 

                 if(side > -1 && !bileaf)
                 {
                     int other = 1 - side ; 
                     tX_min[side] = isect[side+LEFT].w + propagate_epsilon ; 
                     CSG_PUSH( csg[other].data, csg[other].curr, ERROR_OVERFLOW, isect[other+LEFT] );
                     unsigned subtree = side == LHS ? POSTORDER_SLICE(i-2*halfNodes, i-halfNodes) : POSTORDER_SLICE(i-halfNodes, i) ;
                     TRANCHE_PUSH( _tranche, _tmin, tranche, ERROR_TRANCHE_OVERFLOW, POSTORDER_SLICE(i, numInternalNodes)        , tmin );
                     TRANCHE_PUSH( _tranche, _tmin, tranche, ERROR_TRANCHE_OVERFLOW, subtree , tX_min[side] );
                     reiterate = true ; 
                     break ; 
                 } 
             }  // side loop


             if(reiterate || abort_) break ;  
             // reiteration needs to get back to tranche loop for subtree traversal 
             // without "return"ing anything

             rflip.x = -right.x ;
             rflip.y = -right.y ;
             rflip.z = -right.z ; 
             rflip.w =  right.w ;

             const float4& result = ctrl < CTRL_LOOP_A ? isect[ctrl] : miss ;   // CTRL_RETURN_*

             int nside = nodeIdx % 2 == 0 ? LHS : RHS ; 
             CSG_PUSH( csg[nside].data, csg[nside].curr, ERROR_RESULT_OVERFLOW, result );

         }  // end for : node traversal within tranche
    }       // end while : tranche


    ierr |= (( lhs.curr != -1 ) ? ERROR_LHS_END_NONEMPTY : 0 ) ;  
    ierr |= (( rhs.curr !=  0)  ? ERROR_RHS_END_EMPTY : 0)  ; 

    if(rhs.curr == 0 && ierr == 0)
    {
         const float4& ret = rhs.data[0] ;  
         if(rtPotentialIntersection( ret.w ))
         {
              shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
              instanceIdentity = identity ;
              rtReportIntersection(0);
         }
    } 

    //rtPrintf("intersect_csg partOffset %u numParts %u numInternalNodes %u primIdx_ %u height %u postorder %llx ierr %x \n", partOffset, numParts, numInternalNodes, primIdx_, height, postorder, ierr );
    if(ierr != 0)
    rtPrintf("intersect_csg primIdx_ %u ierr %4x   \n", primIdx_, ierr );

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


    enum { MISS, LEFT, RIGHT, RFLIP } ;
    float4 isect[4] ;
    isect[MISS]       =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[LEFT]       =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[RIGHT]      =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[RFLIP]      =  make_float4(0.f, 0.f, 1.f, 0.f);

    float4& left  = isect[LEFT];
    float4& right = isect[RIGHT];
    float4& rflip = isect[RFLIP];

    float tA_min = ray.tmin ; // formerly propagate_epsilon and before that 0.f
    float tB_min = ray.tmin ;

    int ctrl = CTRL_UNDEFINED ; 
    enum { LIVE_DONE = 0, LIVE_A = 0x1 << 0, LIVE_B = 0x1 << 1 } ;
    int live = LIVE_A | LIVE_B ; 
 

    int loop(-1) ;  
    while( live && loop < 10 )
    {
        loop++ ; 

        if(live & LIVE_A) intersect_part( a_partIdx , tA_min, left  ) ;
        if(live & LIVE_B) intersect_part( b_partIdx , tB_min, right ) ;

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

        ctrl = boolean_ctrl_packed_lookup( operation, a_state, b_state, left.w <= right.w );

        switch(ctrl)
        {
            case CTRL_LOOP_A: tA_min = left.w + propagate_epsilon ; live = LIVE_A     ; break ;
            case CTRL_LOOP_B: tB_min = right.w + propagate_epsilon ; live = LIVE_B    ; break ;
            default:                                                 live = LIVE_DONE ; break ;  
        }
    } 


    rflip.x = -right.x ;
    rflip.y = -right.y ;
    rflip.z = -right.z ; 
    rflip.w =  right.w ;


    if(ctrl < 4)
    {
       const float4& ret = isect[ctrl] ; 

        if(rtPotentialIntersection(ret.w))
        {
            shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
            instanceIdentity = identity ;

#ifdef BOOLEAN_DEBUG
            instanceIdentity.x = loop  ; 
#endif
            rtReportIntersection(0);
        }
    }
}



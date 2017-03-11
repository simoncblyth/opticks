#pragma once



__device__ __forceinline__
unsigned long getBitField(unsigned long val, int pos, int len) {
  unsigned long ret;
  asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
  return ret;
}



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


#define CSG_STACK_SIZE 15
#define TRANCHE_STACK_SIZE 4

#define POSTORDER_NODE(postorder, i) (((postorder) & (0xFull << (i)*4 )) >> (i)*4 )
#define POSTORDER_NODE_BFE(postorder, i) (getBitField((postorder), (i)*4, 4))




/**
   stack indices
                       -1 : empty
                        0 : one item
                        1 : two items
                       ..
                 SIZE - 1 : SIZE items, full stack

**/


#define CSG_PUSH(_stack, stack, ERR, val) \
             { \
                 if((stack) >= CSG_STACK_SIZE - 1 ) \
                 {  \
                     ierr |= (ERR) ; \
                     break ;  \
                 }  \
                 (stack)++ ;   \
                 (_stack)[(stack)] = val ; \
             } \


#define CSG_PUSH_(ERR, val, idx) \
             { \
                 if(csg.curr >= CSG_STACK_SIZE - 1 ) \
                 {  \
                     ierr |= (ERR) ; \
                     break ;  \
                 }  \
                 csg.curr++ ;   \
                 csg.data[csg.curr] = (val) ; \
                 csg.idx[csg.curr] = (idx) ; \
             } \



#define CSG_POP(_stack, stack, ERR, ret ) \
             { \
                if((stack) < 0) \
                {  \
                    ierr |= (ERR)  ;\
                    break ; \
                }  \
                (ret) = (_stack)[(stack)] ;  \
                (stack)-- ;    \
             } \

#define CSG_POP_(ERR, ret, idx ) \
             { \
                if(csg.curr < 0) \
                {  \
                    ierr |= (ERR)  ;\
                    break ; \
                }  \
                (ret) = csg.data[csg.curr] ;  \
                (idx) = csg.idx[csg.curr] ;  \
                csg.curr-- ;    \
             } \













// pop without returning data
#define CSG_POP0(_stack, stack, ERR ) \
             { \
                if((stack) < 0) \
                {  \
                    ierr |= (ERR)  ;\
                    break ; \
                }  \
                (stack)-- ;    \
             } \
 

#define CSG_POP0_(ERR ) \
             { \
                if(csg.curr < 0) \
                {  \
                    ierr |= (ERR)  ;\
                    break ; \
                }  \
                csg.curr-- ;    \
             } \
 
 


 

#define CSG_CLASSIFY( ise, dir, tmin )   (fabsf((ise).w) > (tmin) ?  ( (ise).x*(dir).x + (ise).y*(dir).y + (ise).z*(dir).z < 0.f ? Enter : Exit ) : Miss )


#define TRANCHE_PUSH0( _stacku, _stackf, stack, ERR, valu, valf ) \
           { \
                if((stack) >= TRANCHE_STACK_SIZE - 1 ) \
                {  \
                    ierr |= ERR ; \
                } \
                else  \
                {  \
                    (stack)++ ; \
                    setByIndex( (_stacku), (stack), (valu) ) ; \
                    setByIndex( (_stackf), (stack), (valf) ) ; \
                } \
           }   


#define TRANCHE_PUSH( _stacku, _stackf, stack, ERR, valu, valf ) \
            { \
                if((stack) >= TRANCHE_STACK_SIZE - 1) \
                {   \
                     ierr |= (ERR) ; \
                     break ; \
                } \
                (stack)++ ; \
                setByIndex( (_stacku), (stack), (valu) ) ; \
                setByIndex( (_stackf), (stack), (valf) ) ; \
           }  


#define TRANCHE_POP0( _stacku, _stackf, stack, valu, valf ) \
         (valf) = getByIndex((_stackf), (stack));  \
         (valu) = getByIndex((_stacku), (stack) ); \
         (stack)-- ; 




#define POSTORDER_SLICE(begin, end) (  (((end) & 0xff) << 8) | ((begin) & 0xff)  )
#define POSTORDER_BEGIN( slice )  ( (slice) & 0xff )
#define POSTORDER_END( slice )  ( (slice) >> 8 )


struct Tranche
{
    float tmin[TRANCHE_STACK_SIZE] ;     // TRANCHE_STACK_SIZE is 4 
    unsigned  slice[TRANCHE_STACK_SIZE] ; 
    int curr ;
};


__device__ int tranche_push(Tranche& tr, const unsigned slice, const float tmin)
{
    if(tr.curr >= TRANCHE_STACK_SIZE - 1) return ERROR_TRANCHE_OVERFLOW ; 
    tr.curr++ ; 
    tr.slice[tr.curr] = slice  ; 
    tr.tmin[tr.curr] = tmin ; 
    return 0 ; 
}

__device__ int tranche_pop(Tranche& tr, unsigned& slice, float& tmin)
{
    if(tr.curr >= TRANCHE_STACK_SIZE - 1) return ERROR_POP_EMPTY  ; 
    slice = tr.slice[tr.curr] ;
    tmin = tr.tmin[tr.curr] ;  
    tr.curr-- ; 
    return 0 ; 
}

__device__ unsigned long long tranche_repr(Tranche& tr)
{
    unsigned long long val = 0 ; 
    if(tr.curr == -1) return val ; 

    unsigned long long c = tr.curr ;
    val |= ((c+1ull)&0xfull)  ;     // count at lsb, contents from msb 
 
    do { 
        unsigned long long x = tr.slice[c] & 0xffff ;
        val |=  x << ((4ull-c-1ull)*16ull) ; 
    } 
    while(c--) ; 

    return val ; 
} 





struct History
{
    enum {
           NUM = 2,     
           SIZE = 64,    // of each carrier     
           NITEM = 16,   // items within each 64bit
           NBITS = 4,     // bits per item    
           MASK  = 0xf 
         } ;  // 
    unsigned long long idx[NUM] ; 
    unsigned long long ctrl[NUM] ; 
    int curr ; 
};


__device__ int history_append( History& h, unsigned idx, unsigned ctrl)
{
    if((h.curr+1) > h.NUM*h.NITEM ) return ERROR_OVERFLOW ; 
    h.curr++ ; 

    int nb = h.curr/h.NITEM  ;                    // target carrier int 
    unsigned long long  ii = h.curr*h.NBITS - h.SIZE*nb ; // bit offset within target 64bit 
    unsigned long long hidx = h.MASK & idx ;
    unsigned long long hctrl = h.MASK & ctrl ;

    h.idx[nb]  |=  hidx << ii   ; 
    h.ctrl[nb] |=  hctrl << ii  ; 

    return 0 ; 
}



struct CSG 
{
   float4 data[CSG_STACK_SIZE] ; 
   unsigned idx[CSG_STACK_SIZE] ; 
   int curr ;
};

__device__ int csg_push(CSG& csg, const float4& isect, unsigned nodeIdx)
{
    if(csg.curr >= CSG_STACK_SIZE - 1) return ERROR_OVERFLOW ; 
    csg.curr++ ; 
    csg.data[csg.curr] = isect ; 
    csg.idx[csg.curr] = nodeIdx ; 
    return 0 ; 
}
__device__ int csg_pop(CSG& csg, float4& isect, unsigned& nodeIdx)
{
    if(csg.curr < 0) return ERROR_POP_EMPTY ;     
    isect = csg.data[csg.curr] ;
    nodeIdx = csg.idx[csg.curr] ;
    csg.idx[csg.curr] = 0u ;   // scrub the idx for debug
    csg.curr-- ; 
    return 0 ; 
}

__device__ int csg_pop0(CSG& csg)
{
    if(csg.curr < 0) return ERROR_POP_EMPTY ;     
    csg.idx[csg.curr] = 0u ;   // scrub the idx for debug
    csg.curr-- ; 
    return 0 ; 
}

__device__ unsigned long long csg_repr(CSG& csg)
{
    unsigned long long val = 0 ; 
    if(csg.curr == -1) return val ; 

    unsigned long long c = csg.curr ; 
    val |= (c+1ull) ;  // count at lsb, contents from msb 
 
    do { 
        unsigned long long x = csg.idx[c] & 0xf ;
        val |=  x << ((16ull-c-1ull)*4ull) ; 
    } 
    while(c--) ; 

    return val ; 
} 





// perfect binary tree assumptions,   2^(h+1) - 1 
#define TREE_HEIGHT(numNodes) ( __ffs((numNodes) + 1) - 2)
#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )
#define TREE_DEPTH(nodeIdx) ( 32 - __clz((nodeIdx)) - 1 )


static __device__
float unsigned_as_float(unsigned u)
{
  union {
    float f;
    unsigned u;
  } v1;

  v1.u = u; 
  return v1.f;
}




//  Below indices are postorder slice flavor, not levelorder
//
//  Height 3 tree, numNodes = 15, halfNodes = 7, root at i = 14 
//
//    upTree       i      : numNodes    14        : 15      =  14:15    
//    leftTree     i - 2h : i - h       14 - 2*7  : 14 - 7  =   0:7       
//    rightTree    i -  h : i           14 -  7   : 14      =   7:14
//
//  NB all nodes including root needs an upTree tranche to evaluate left and right 


/*
Hmm seems cannot do recursive within an intersect program anyhow ??

libc++abi.dylib: terminating with uncaught exception of type optix::Exception: 
Parse error (Details: Function "RTresult _rtProgramCreateFromPTXFile(RTcontext, const char *, const char *, RTprogram *)" 
caught exception: /usr/local/opticks/installcache/PTX/OptiXRap_generated_hemi-pmt.cu.ptx: 
error: Recursion detected in function 
_Z15recursive_csg_rjjjf( file /usr/local/opticks/installcache/PTX/OptiXRap_generated_hemi-pmt.cu.ptx ), 
cannot inline. [5308892], [5308892])

*/

__device__
float4 recursive_csg_r( unsigned partOffset, unsigned numInternalNodes, unsigned nodeIdx, float tmin )
{
    unsigned leftIdx = nodeIdx*2 ; 
    unsigned rightIdx = leftIdx + 1 ; 
    bool bileaf = leftIdx > numInternalNodes ; 
    enum { LEFT=0, RIGHT=1 };

    float4 isect[2] ;  
    isect[LEFT] = make_float4(0.f, 0.f, 0.f, 0.f) ;  
    isect[RIGHT] = make_float4(0.f, 0.f, 0.f, 0.f) ;  
    if(bileaf)
    {
        intersect_part( partOffset+leftIdx-1, tmin, isect[LEFT] );
        intersect_part( partOffset+rightIdx-1, tmin, isect[RIGHT] );
    }  
    else
    {
        isect[LEFT]  = recursive_csg_r( partOffset, numInternalNodes, leftIdx, tmin);
        isect[RIGHT] = recursive_csg_r( partOffset, numInternalNodes, rightIdx, tmin);
    } 
    quad q1 ; 
    q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];      // (nodeIdx-1) as 1-based
    OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

    IntersectionState_t x_state[2] ; 
    x_state[LEFT]  = CSG_CLASSIFY( isect[LEFT], ray.direction, tmin );
    x_state[RIGHT] = CSG_CLASSIFY( isect[RIGHT], ray.direction, tmin );

    float x_tmin[2] ;
    x_tmin[LEFT] = tmin ; 
    x_tmin[RIGHT] = tmin ; 

    int ctrl = boolean_ctrl_packed_lookup( operation, x_state[LEFT], x_state[RIGHT], isect[LEFT].w <= isect[RIGHT].w ) ;

    if(ctrl >= CTRL_LOOP_A)  // not ready to return yet 
    {
        int side = ctrl - CTRL_LOOP_A ;
        int loop = -1 ; 
        while(side > -1 && loop < 10)
        {
            loop += 1 ; 
            x_tmin[side] = isect[side].w + propagate_epsilon ; 
            if(bileaf)
            {
                intersect_part( partOffset+leftIdx+side-1, x_tmin[side], isect[LEFT+side] );
            }
            else
            {
                isect[LEFT+side] = recursive_csg_r( partOffset, numInternalNodes, leftIdx+side, x_tmin[side]);
            }
            x_state[LEFT+side] = CSG_CLASSIFY( isect[LEFT+side], ray.direction, x_tmin[side] );
            ctrl = boolean_ctrl_packed_lookup( operation, x_state[LEFT], x_state[RIGHT], isect[LEFT].w <= isect[RIGHT].w ) ;
            side = ctrl - CTRL_LOOP_A  ; 
        }
    }
    float4 result = ctrl == CTRL_RETURN_MISS ?  make_float4(0.f, 0.f, 0.f, 0.f ) : ( ctrl == CTRL_RETURN_A ? isect[LEFT] : isect[RIGHT] ) ;
    if(ctrl == CTRL_RETURN_FLIP_B)
    {
        result.x = -result.x ;     
        result.y = -result.y ;     
        result.z = -result.z ;     
    }
    return result ;
}
 


static __device__
void recursive_csg( const uint4& prim, const uint4& identity )
{
    unsigned partOffset = prim.x ; 
    unsigned numParts   = prim.y ;
    unsigned fullHeight = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
    unsigned numInternalNodes = TREE_NODES(fullHeight-1) ;
    unsigned rootIdx = 1 ; 

    float4 ret = recursive_csg_r( partOffset, numInternalNodes, rootIdx, ray.tmin ); 
    if(rtPotentialIntersection( fabsf(ret.w) ))
    {
        shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
        instanceIdentity = identity ;
        rtReportIntersection(0);
    } 
}




 


static __device__
void evaluative_csg( const uint4& prim, const uint4& identity )
{
    unsigned partOffset = prim.x ; 
    unsigned numParts   = prim.y ;
    unsigned primIdx_   = prim.z ; 
    unsigned fullHeight = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 

    //rtPrintf("evaluative_csg primIdx_ %u numParts %u perfect tree fullHeight %u  \n",primIdx_, numParts, fullHeight ) ; 
    if(fullHeight > 3)
    {
        rtPrintf("evaluative_csg primIdx_ %u numParts %u perfect tree fullHeight %u exceeds current limit\n", primIdx_, numParts, fullHeight ) ;
        return ; 
    } 
    unsigned height = fullHeight - 1 ;
    unsigned numInternalNodes = TREE_NODES(height) ;
    unsigned numNodes         = TREE_NODES(fullHeight) ;      

    const unsigned long long postorder_sequence[4] = { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;
    unsigned long long postorder = postorder_sequence[fullHeight] ; 
    //rtPrintf("evaluative_csg primIdx_ %u fullHeight %u numInternalNodes %u numNodes  %u postorder %16llx  \n", primIdx_, fullHeight, numInternalNodes, numNodes, postorder );

    int ierr = 0 ;  
    bool verbose = false ; 

    History hist ; 
    hist.curr = -1 ; 
    hist.ctrl[0] = 0 ;
    hist.ctrl[1] = 0 ;
    hist.idx[0] = 0 ;
    hist.idx[1] = 0 ;

    Tranche tr ; 
    tr.curr = -1 ;
    tranche_push( tr, POSTORDER_SLICE(0, numNodes), ray.tmin );

    CSG csg ;  
    csg.curr = -1 ;

    // make global and add prevs for debug
    unsigned nodeIdx = 0 ; 
    unsigned prevIdx = 0 ; 
    int ctrl = -1 ; 
    int prevCtrl = -1 ; 
    int tloop = -1 ; 


    while (tr.curr > -1)
    {
        tloop++ ; 
        unsigned slice ; 
        float tmin ; 
        ierr = tranche_pop(tr, slice, tmin );
        if(ierr) break ; 

        unsigned begin = POSTORDER_BEGIN(slice);
        unsigned end   = POSTORDER_END(slice);

        unsigned beginIdx = POSTORDER_NODE(postorder, begin);   
        unsigned endIdx = POSTORDER_NODE(postorder, end - 1);   

        if(verbose)
        rtPrintf("[%5d](trav) nodeIdx %2d csg.curr %2d csg_repr %16llx tr_repr %16llx tloop %2d prevIdx %d [%x:%x] (%2u->%2u) %7.3f \n", 
                           launch_index.x, 
                           nodeIdx,
                           csg.curr,
                           csg_repr(csg), 
                           tranche_repr(tr),
                           tloop,  
                           prevIdx,
                           begin,
                           end,
                           POSTORDER_NODE(postorder, begin),
                           POSTORDER_NODE(postorder, end-1),
                           tmin 
                              );


        for(unsigned i=begin ; i < end ; i++)
        {
            prevIdx = nodeIdx ; 
            nodeIdx = POSTORDER_NODE(postorder, i) ;

            int depth = TREE_DEPTH(nodeIdx) ;
            unsigned subNodes = TREE_NODES(fullHeight-depth) ;
            unsigned halfNodes = (subNodes - 1)/2 ; 
            bool primitive = nodeIdx > numInternalNodes  ;  // TODO: use partBuffer content for empty handling

            quad q1 ; 
            q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];      // (nodeIdx-1) as 1-based
            OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

            //if(prevIdx == nodeIdx)

            if(verbose)
            rtPrintf("[%5d](visi) nodeIdx %2d csg.curr %2d csg_repr %16llx tr_repr %16llx tloop %2d  operation %d primitive %d halfNodes %2d depth %d prevIdx %d \n", 
                           launch_index.x, 
                           nodeIdx,
                           csg.curr,
                           csg_repr(csg), 
                           tranche_repr(tr),
                           tloop,  
                           operation,
                           primitive,
                           halfNodes,
                           depth,
                           prevIdx
                              );

            if(primitive)
            {
                float4 isect = make_float4(0.f, 0.f, 0.f, 0.f) ;  
                intersect_part( partOffset+nodeIdx-1, tmin, isect );
                isect.w = copysignf( isect.w, nodeIdx % 2 == 0 ? -1.f : 1.f );  // hijack t signbit, to record the side, LHS -ve

                ierr = csg_push(csg, isect, nodeIdx ); 
                if(ierr) break ; 
            /*
                rtPrintf("[%5d](prim) nodeIdx %2d csg.curr %2d csg_repr %16llx tr_repr %16llx  (%5.2f,%5.2f,%5.2f,%7.3f)  \n",
                     launch_index.x, 
                     nodeIdx, 
                     csg.curr,
                     csg_repr(csg),
                     tranche_repr(tr),
                     isect.x, isect.y, isect.z, isect.w 
                     );
             */
            }
            else
            {
                if(csg.curr < 1)  // curr 1 : 2 items 
                {
                    rtPrintf("[%5d]evaluative_csg ERROR_POP_EMPTY nodeIdx %4d operation %d csg.curr %d \n", launch_index.x, nodeIdx, operation, csg.curr );
                    ierr |= ERROR_POP_EMPTY ; 
                    break ; 
                }
                bool firstLeft = signbit(csg.data[csg.curr].w) ;
                bool secondLeft = signbit(csg.data[csg.curr-1].w) ;

                if(!(firstLeft ^ secondLeft))
                {
                    rtPrintf("[%5d]evaluative_csg ERROR_XOR_SIDE nodeIdx %4d operation %d tl %10.3f tr %10.3f sl %d sr %d \n", launch_index.x, nodeIdx, operation, csg.data[csg.curr].w, csg.data[csg.curr-1].w, firstLeft, secondLeft );
                    ierr |= ERROR_XOR_SIDE ; 
                    break ; 
                }
                int left  = firstLeft ? csg.curr   : csg.curr-1 ;
                int right = firstLeft ? csg.curr-1 : csg.curr   ; 

                IntersectionState_t l_state = CSG_CLASSIFY( csg.data[left], ray.direction, tmin );
                IntersectionState_t r_state = CSG_CLASSIFY( csg.data[right], ray.direction, tmin );

                float t_left  = fabsf( csg.data[left].w );
                float t_right = fabsf( csg.data[right].w );

                prevCtrl = ctrl ; 
                ctrl = boolean_ctrl_packed_lookup( operation, l_state, r_state, t_left <= t_right ) ;
                history_append( hist, nodeIdx, ctrl ); 

                enum { UNDEFINED=0, CONTINUE=1, BREAK=2 } ;

                int act = UNDEFINED ; 

                if(ctrl < CTRL_LOOP_A) // "returning" with a push 
                {
                    float4 result = ctrl == CTRL_RETURN_MISS ?  make_float4(0.f, 0.f, 0.f, 0.f ) : ( ctrl == CTRL_RETURN_A ? csg.data[left] : csg.data[right] ) ;
                    if(ctrl == CTRL_RETURN_FLIP_B)
                    {
                        result.x = -result.x ;     
                        result.y = -result.y ;     
                        result.z = -result.z ;     
                    }
                    result.w = copysignf( result.w , nodeIdx % 2 == 0 ? -1.f : 1.f );

                    ierr = csg_pop0(csg); if(ierr) break ;
                    ierr = csg_pop0(csg); if(ierr) break ;
                    ierr = csg_push(csg, result, nodeIdx );  if(ierr) break ;

                    act = CONTINUE ;  
                }
                else
                {                 
                    int loopside  = ctrl == CTRL_LOOP_A ? left : right ;    
                    int otherside = ctrl == CTRL_LOOP_A ? right : left ;  

                    unsigned leftIdx = 2*nodeIdx ; 
                    unsigned rightIdx = leftIdx + 1; 
                    unsigned otherIdx = ctrl == CTRL_LOOP_A ? rightIdx : leftIdx ; 

                    float tminAdvanced = fabsf(csg.data[loopside].w) + propagate_epsilon ;
                    float4 other = csg.data[otherside] ;   

                    ierr = csg_pop0(csg);                   if(ierr) break ;
                    ierr = csg_pop0(csg);                   if(ierr) break ;
                    ierr = csg_push(csg, other, otherIdx ); if(ierr) break ;

                    // looping is effectively backtracking, pop both and put otherside back

                    unsigned upTree    = POSTORDER_SLICE(i, numNodes);
                    unsigned leftTree  = POSTORDER_SLICE(i-2*halfNodes, i-halfNodes) ;
                    unsigned rightTree = POSTORDER_SLICE(i-halfNodes, i) ;
                    unsigned loopTree  = ctrl == CTRL_LOOP_A ? leftTree : rightTree  ;

                    ierr = tranche_push( tr, upTree, tmin );           if(ierr) break ;
                    ierr = tranche_push( tr, loopTree, tminAdvanced ); if(ierr) break ; 

                    act = BREAK  ;  
                }             // "return" or "recursive call" 

                if(verbose)
                rtPrintf("[%5d](ctrl) nodeIdx %2d csg.curr %2d csg_repr %16llx tr_repr %16llx ctrl %d     tloop %2d (%2d->%2d) operation %d tlr (%10.3f,%10.3f) \n", 
                           launch_index.x, 
                           nodeIdx,
                           csg.curr,
                           csg_repr(csg), 
                           tranche_repr(tr),
                           ctrl,
                           tloop,  
                           beginIdx,
                           endIdx,  
                           operation, 
                           t_left, 
                           t_right
                              );




                if(act == BREAK) break ; 
            }                 // "primitive" or "operation"
        }                     // node traversal 
        if(ierr) break ; 
     }                       // subtree tranches



    ierr |= (( csg.curr !=  0)  ? ERROR_END_EMPTY : 0)  ; 

    //if(ierr == 0)   // ideally, but for now incude error returns, to see where the problems are
    if(csg.curr == 0)  
    {
         const float4& ret = csg.data[0] ;   
/*
         rtPrintf("[%5d]evaluative_csg ierr %4x ray.origin (%10.3f,%10.3f,%10.3f) ray.direction (%10.3f,%10.3f,%10.3f) ret (%5.2f,%5.2f,%5.3f,%7.3f) \n",
               launch_index.x, ierr, 
               ray.origin.x, ray.origin.y, ray.origin.z,
               ray.direction.x, ray.direction.y, ray.direction.z,
               ret.x, ret.y, ret.z, ret.w );
*/  
             
         if(rtPotentialIntersection( fabsf(ret.w) ))
         {
              shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
              instanceIdentity = identity ;
#ifdef BOOLEAN_DEBUG
              instanceIdentity.x = ierr > 0 ? 1 : 0 ;   // used for visualization coloring  
              instanceIdentity.y = ierr ; 
              // instanceIdentity.z is used for boundary passing, hijacking prevents photon visualization
              instanceIdentity.w = tloop ; 
#endif
              rtReportIntersection(0);
         }
    } 

#ifdef BOOLEAN_DEBUG
        /*
        rtPrintf("[%5d]evaluative_csg ERROR ierr %4x prevNode %2d nodeIdx %2d csg.curr %d tranche %d  ray.direction (%10.3f,%10.3f,%10.3f) ray.origin (%10.3f,%10.3f,%10.3f)   \n",
              launch_index.x, ierr, prevNode, nodeIdx, csg.curr, tranche,
              ray.direction.x, ray.direction.y, ray.direction.z,
              ray.origin.x, ray.origin.y, ray.origin.z
              );
        */

     //if(ierr != 0)
     //if(ierr == 0x1000 && csg.curr != 0)
     //if(ierr == 0x6000 && csg.curr != 1)

        if(verbose || ierr !=0)
        rtPrintf("[%5d](DONE) nodeIdx %2d csg.curr %2d csg_repr %16llx tr_repr %16llx IERR %6x hcur %2d hi %16llx:%16llx hc %16llx:%16llx \n",
                           launch_index.x, 
                           nodeIdx,
                           csg.curr,
                           csg_repr(csg), 
                           tranche_repr(tr),
                           ierr, 
                           hist.curr,
                           hist.idx[1], 
                           hist.idx[0], 
                           hist.ctrl[1],
                           hist.ctrl[0]
                             );





#endif
}



static __device__
void intersect_csg( const uint4& prim, const uint4& identity )
{
/**

  * postorder traversal means that have always 
    visited left and right subtrees before visiting a node

  * a slavish python translation of this is in dev/csg/slavish.py 

  * postorder_sequence for four tree heights were prepared by  
    opticks/dev/csg/node.py:Node.postOrderSequence
  
  * the sequence contains 1-based levelorder indices(nodeIdx) in right to left postorder

  * left,right child of nodeIdx are at (nodeIdx*2, nodeIdx*2+1)

  * for non-perfect trees, the height means the maximum 

**/

    const unsigned long long postorder_sequence[4] = { 0x1ull, 0x132ull, 0x1376254ull, 0x137fe6dc25ba498ull } ;
    //const unsigned long long pseq4[4] = { 0x103070f1f1e0eull,0x1d1c060d1b1a0c19ull,0x1802050b17160a15ull,0x1404091312081110ull } ;


    int ierr = 0 ;  
    int loop = -1 ; 

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

#ifdef BOOLEAN_DEBUG
    int tloop = -1 ; 
#endif

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

    TRANCHE_PUSH0( _tranche, _tmin, tranche, ERROR_TRANCHE_OVERFLOW, POSTORDER_SLICE(0, numInternalNodes), ray.tmin );

    while (tranche >= 0)
    {
#ifdef BOOLEAN_DEBUG
         tloop += 1 ;
#endif 
         float   tmin ;
         unsigned tmp ;
         TRANCHE_POP0( _tranche, _tmin, tranche,  tmp, tmin );
         unsigned begin = POSTORDER_BEGIN( tmp );
         unsigned end   = POSTORDER_END( tmp );

         for(unsigned i=begin ; i < end ; i++)
         {
             // XXidx are 1-based levelorder perfect tree indices
             unsigned nodeIdx = POSTORDER_NODE_BFE(postorder, i) ;   
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
             tX_min[LHS] = tmin ;
             tX_min[RHS] = tmin ;

             IntersectionState_t x_state[2] ; 

             if(bileaf) // op-left-right leaves
             {
                 left.w = 0.f ;   // reusing the same storage so clear ahead
                 right.w = 0.f ; 
                 intersect_part( partOffset+leftIdx-1 , tX_min[LHS], left  ) ;
                 intersect_part( partOffset+rightIdx-1 , tX_min[RHS], right  ) ;
             }
             else       //  op-op-op
             {
                 CSG_POP( lhs.data, lhs.curr, ERROR_LHS_POP_EMPTY, left );
                 CSG_POP( rhs.data, rhs.curr, ERROR_RHS_POP_EMPTY, right );
             }
 
             x_state[LHS] = CSG_CLASSIFY( left , ray.direction, tX_min[LHS] ) ;
             x_state[RHS] = CSG_CLASSIFY( right, ray.direction, tX_min[RHS] ) ;

             int ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], left.w <= right.w );
             int side = ctrl - CTRL_LOOP_A ;   // CTRL_LOOP_A,CTRL_LOOP_B -> LHS, RHS   looper side

             bool reiterate = false ; 
             loop = -1  ;  
             while( side > -1 && loop < 10 )
             {
                 loop++ ; 
                 float4& _side = isect[side+LEFT] ; 

                 tX_min[side] = _side.w + propagate_epsilon ;  // classification as well as intersect needs the advance
            
                 if(bileaf)
                 {
                      intersect_part( partOffset+leftIdx+side-1 , tX_min[side], _side  ) ; // tmin advance
                 }
                 else
                 {
                      CSG_POP( csg[side].data, csg[side].curr, ERROR_POP_EMPTY, _side ); // faux recursive call
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
                     TRANCHE_PUSH( _tranche, _tmin, tranche, ERROR_TRANCHE_OVERFLOW, POSTORDER_SLICE(i, numInternalNodes), tmin );
                     TRANCHE_PUSH( _tranche, _tmin, tranche, ERROR_TRANCHE_OVERFLOW, subtree , tX_min[side] );
                     reiterate = true ; 
                     break ; 
                 } 
             }  // side loop


             if(reiterate || ierr ) break ;  
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
         if(ierr) break ;
    }       // end while : tranche


    ierr |= (( lhs.curr != -1 ) ? ERROR_LHS_END_NONEMPTY : 0 ) ;  
    ierr |= (( rhs.curr !=  0)  ? ERROR_RHS_END_EMPTY : 0)  ; 

    if(rhs.curr == 0 || lhs.curr == 0)
    {
         const float4& ret = rhs.curr == 0 ? rhs.data[0] : lhs.data[0] ;   // <-- should always be rhs, accept lhs for debug
         if(rtPotentialIntersection( ret.w ))
         {
              shading_normal = geometric_normal = make_float3(ret.x, ret.y, ret.z) ;
              instanceIdentity = identity ;
#ifdef BOOLEAN_DEBUG
              instanceIdentity.x = ierr != 0  ? 1 : instanceIdentity.x ; 
              instanceIdentity.y = tloop == 1 ? 1 : instanceIdentity.y ; 
              instanceIdentity.z = tloop > 1  ? 1 : instanceIdentity.z ; 
#endif
              rtReportIntersection(0);
         }
    } 

    //rtPrintf("intersect_csg partOffset %u numParts %u numInternalNodes %u primIdx_ %u height %u postorder %llx ierr %x \n", partOffset, numParts, numInternalNodes, primIdx_, height, postorder, ierr );
    if(ierr != 0)
    rtPrintf("intersect_csg primIdx_ %u ierr %4x tloop %3d launch_index (%5d,%5d) li.x(26) %2d ray.direction (%10.3f,%10.3f,%10.3f) ray.origin (%10.3f,%10.3f,%10.3f)   \n",
          primIdx_, ierr, tloop, launch_index.x, launch_index.y,  launch_index.x % 26,
          ray.direction.x, ray.direction.y, ray.direction.z,
          ray.origin.x, ray.origin.y, ray.origin.z
      );

}   // intersect_csg








static __device__
void intersect_boolean_triplet( const uint4& prim, const uint4& identity )
{
    unsigned partOffset = prim.x ; 
    //unsigned primIdx_   = prim.z ; 

    unsigned nodeIdx = 1 ;    
    unsigned leftIdx = nodeIdx*2 ;      
    unsigned rightIdx = nodeIdx*2 + 1 ;  

    quad q1 ; 
    q1.f = partBuffer[4*(partOffset+nodeIdx-1)+1];
    OpticksCSG_t operation = (OpticksCSG_t)q1.u.w ;

    //rtPrintf("intersect_boolean primIdx_:%u n:%u a:%u b:%u operation:%u \n", primIdx_, n_partIdx, a_partIdx, b_partIdx, operation );

    enum { LHS, RHS };
    enum { MISS, LEFT, RIGHT, RFLIP } ;
    float4 isect[4] ;
    isect[MISS]       =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[LEFT]       =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[RIGHT]      =  make_float4(0.f, 0.f, 1.f, 0.f);
    isect[RFLIP]      =  make_float4(0.f, 0.f, 1.f, 0.f);

    float4& left  = isect[LEFT];
    float4& right = isect[RIGHT];
    float4& rflip = isect[RFLIP];

    float tX_min[2] ; 
    tX_min[LHS] = ray.tmin ; 
    tX_min[RHS] = ray.tmin ; 

    left.w = 0.f ;   // reusing the same storage so clear ahead
    right.w = 0.f ; 
    intersect_part( partOffset+leftIdx-1 , tX_min[LHS], left  ) ;
    intersect_part( partOffset+rightIdx-1 , tX_min[RHS], right  ) ;

    IntersectionState_t x_state[2] ;

    x_state[LHS] = CSG_CLASSIFY( left, ray.direction, tX_min[LHS] );
    x_state[RHS] = CSG_CLASSIFY( right, ray.direction, tX_min[RHS] );

    int ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], left.w <= right.w );
    int side = ctrl - CTRL_LOOP_A ; 

    int loop(-1);
    while( side > -1 && loop < 10)
    {
        loop++ ;
        float4& _side = isect[side+LEFT] ;
        tX_min[side] = _side.w + propagate_epsilon ;
        intersect_part( partOffset+leftIdx+side-1 , tX_min[side], _side  ) ; // tmin advanced intersect

        x_state[side] = CSG_CLASSIFY( _side, ray.direction, tX_min[side] );
        ctrl = boolean_ctrl_packed_lookup( operation, x_state[LHS], x_state[RHS], left.w <= right.w );
        side = ctrl - CTRL_LOOP_A ;
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



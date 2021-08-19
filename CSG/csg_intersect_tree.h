#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define TREE_FUNC __forceinline__ __device__
#else
#    define TREE_FUNC
#endif

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <cmath>  // signbit
using std::signbit ; 
#endif

#include "csg_error.h"
#include "csg_tranche.h"
#include "csg_stack.h"
#include "csg_postorder.h"
#include "csg_pack.h"
#include "csg_classify.h"


TREE_FUNC
bool intersect_tree( float4& isect, int numNode, const CSGNode* node, const float4* plan0, const qat4* itra0, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    unsigned height = TREE_HEIGHT(numNode) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
    float propagate_epsilon = 0.0001f ;  // ? 
    int ierr = 0 ;  

    LUT lut ; 
    Tranche tr ; 
    tr.curr = -1 ;

    unsigned fullTree = PACK4(0,0, 1 << height, 0 ) ;  // leftmost: 1<<height,  root:1>>1 = 0 ("parent" of root)  
 
#ifdef DEBUG
    printf("//intersect_tree.h numNode %d height %d fullTree(hex) %x \n", numNode, height, fullTree );
#endif

    tranche_push( tr, fullTree, t_min );

    CSG_Stack csg ;  
    csg.curr = -1 ;
    int tloop = -1 ; 

    while (tr.curr > -1)
    {
        tloop++ ; 
        unsigned slice ; 
        float tmin ; 

        ierr = tranche_pop(tr, slice, tmin );
        if(ierr) break ; 

        // beginIdx, endIdx are 1-based level order tree indices, root:1 leftmost:1<<height 
        unsigned beginIdx = UNPACK4_2(slice);
        unsigned endIdx   = UNPACK4_3(slice);

        unsigned nodeIdx = beginIdx ; 
        while( nodeIdx != endIdx )
        {
            unsigned depth = TREE_DEPTH(nodeIdx) ;
            unsigned elevation = height - depth ; 

            const CSGNode* nd = node + nodeIdx - 1 ; 
            OpticksCSG_t typecode = (OpticksCSG_t)nd->typecode() ;
#ifdef DEBUG
            printf("//intersect_tree.h nodeIdx %d CSG::Name %10s depth %d elevation %d \n", nodeIdx, CSG::Name(typecode), depth, elevation ); 
#endif
            if( typecode == CSG_ZERO )
            {
                nodeIdx = POSTORDER_NEXT( nodeIdx, elevation ) ;
                continue ; 
            }
            bool primitive = typecode >= CSG_SPHERE ; 
#ifdef DEBUG
            printf("//intersect_tree.h nodeIdx %d primitive %d \n", nodeIdx, primitive ); 
#endif
            if(primitive)
            {
                float4 nd_isect = make_float4(0.f, 0.f, 0.f, 0.f) ;  

                intersect_node( nd_isect, nd, plan0, itra0, tmin, ray_origin, ray_direction );

                nd_isect.w = copysignf( nd_isect.w, nodeIdx % 2 == 0 ? -1.f : 1.f );  // hijack t signbit, to record the side, LHS -ve

#ifdef DEBUG
                printf("//intersect_tree.h nodeIdx %d primitive %d nd_isect (%10.4f %10.4f %10.4f %10.4f) \n", nodeIdx, primitive, nd_isect.x, nd_isect.y, nd_isect.z, nd_isect.w ); 
#endif

                ierr = csg_push(csg, nd_isect, nodeIdx ); 
                if(ierr) break ; 
            }
            else
            {
                if(csg.curr < 1)  // curr 1 : 2 items 
                {
#ifdef DEBUG
                    printf("//intersect_tree.h ERROR_POP_EMPTY nodeIdx %4d typecode %d csg.curr %d \n", nodeIdx, typecode, csg.curr );
#endif
                    ierr |= ERROR_POP_EMPTY ; 
                    break ; 
                }
                bool firstLeft = signbit(csg.data[csg.curr].w) ;
                bool secondLeft = signbit(csg.data[csg.curr-1].w) ;

                if(!(firstLeft ^ secondLeft))
                {
#ifdef DEBUG
                    printf("//intersect_tree.h ERROR_XOR_SIDE nodeIdx %4d typecode %d tl %10.3f tr %10.3f sl %d sr %d \n",
                             nodeIdx, typecode, csg.data[csg.curr].w, csg.data[csg.curr-1].w, firstLeft, secondLeft );
#endif
                    ierr |= ERROR_XOR_SIDE ; 
                    break ; 
                }
                int left  = firstLeft ? csg.curr   : csg.curr-1 ;
                int right = firstLeft ? csg.curr-1 : csg.curr   ; 

                IntersectionState_t l_state = CSG_CLASSIFY( csg.data[left],  ray_direction, tmin );
                IntersectionState_t r_state = CSG_CLASSIFY( csg.data[right], ray_direction, tmin );

                float t_left  = fabsf( csg.data[left].w );
                float t_right = fabsf( csg.data[right].w );

                bool leftIsCloser = t_left <= t_right ;
#ifdef DEBUG
                printf("//intersect_tree.h nodeIdx %4d left %d right %d l_state %d r_state %d  (0/1/2:Enter/Exit/Miss) t_left %10.4f t_right %10.4f leftIsCloser %d \n", 
                       nodeIdx,left,right,l_state,r_state, t_left, t_right, leftIsCloser ); 
#endif
                // complements (signalled by -0.f) cannot Miss, only Exit, see opticks/notes/issues/csg_complement.rst 
                // these are only valid (and only needed) for misses 

                bool l_complement = signbit(csg.data[left].x) ;
                bool r_complement = signbit(csg.data[right].x) ;

                bool l_complement_miss = l_state == State_Miss && l_complement ;
                bool r_complement_miss = r_state == State_Miss && r_complement ;

                if(r_complement_miss)
                {
                    r_state = State_Exit ; 
                    leftIsCloser = true ; 
                } 

                if(l_complement_miss)
                {
                    l_state = State_Exit ; 
                    leftIsCloser = false ; 
                } 

                int ctrl = lut.lookup( typecode , l_state, r_state, leftIsCloser ) ;

                Action_t act = UNDEFINED ; 

                if(ctrl < CTRL_LOOP_A) // CTRL_RETURN_MISS/CTRL_RETURN_A/CTRL_RETURN_B/CTRL_RETURN_FLIP_B "returning" with a push 
                {
                    float4 result = ctrl == CTRL_RETURN_MISS ?  make_float4(0.f, 0.f, 0.f, 0.f ) : csg.data[ctrl == CTRL_RETURN_A ? left : right] ;
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
                else   //   CTRL_LOOP_A/CTRL_LOOP_B
                {                 
                    int loopside  = ctrl == CTRL_LOOP_A ? left : right ;    
                    int otherside = ctrl == CTRL_LOOP_A ? right : left ;  

                    unsigned leftIdx = 2*nodeIdx ; 
                    unsigned rightIdx = leftIdx + 1; 
                    unsigned otherIdx = ctrl == CTRL_LOOP_A ? rightIdx : leftIdx ; 

                    float tminAdvanced = fabsf(csg.data[loopside].w) + propagate_epsilon ;
                    float4 other = csg.data[otherside] ;  // need tmp as pop about to invalidate indices

                    ierr = csg_pop0(csg);                   if(ierr) break ;
                    ierr = csg_pop0(csg);                   if(ierr) break ;
                    ierr = csg_push(csg, other, otherIdx ); if(ierr) break ;

                    // looping is effectively backtracking, pop both and put otherside back

                    unsigned endTree   = PACK4(  0,  0,  nodeIdx,  endIdx  );
                    unsigned leftTree  = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
                    unsigned rightTree = PACK4(  0,  0,  rightIdx << (elevation-1), nodeIdx );

                    unsigned loopTree  = ctrl == CTRL_LOOP_A ? leftTree : rightTree  ;

#ifdef DEBUG
                    printf("//intersect_tree.h nodeIdx %2d height %2d depth %2d elevation %2d endTree %8x leftTree %8x rightTree %8x \n",
                              nodeIdx,
                              height,
                              depth,
                              elevation,
                              endTree, 
                              leftTree,
                              rightTree);
#endif

                    ierr = tranche_push( tr, endTree, tmin );          if(ierr) break ;
                    ierr = tranche_push( tr, loopTree, tminAdvanced ); if(ierr) break ; 

                    act = BREAK  ;  
                }                      // "return" or "recursive call" 
                if(act == BREAK) break ; 
            }                          // "primitive" or "operation"
            nodeIdx = POSTORDER_NEXT( nodeIdx, elevation ) ;
        }                     // node traversal 
        if(ierr) break ; 
    }                        // subtree tranches

    ierr |= (( csg.curr !=  0)  ? ERROR_END_EMPTY : 0)  ; 

#ifdef DEBUG
    printf("//intersect_tree.h ierr %d csg.curr %d \n", ierr, csg.curr ); 
#endif
    if(csg.curr == 0)  
    {
        const float4& ret = csg.data[0] ;   
        isect.x = ret.x ; 
        isect.y = ret.y ; 
        isect.z = ret.z ; 
        isect.w = ret.w ; 
    }
    return isect.w > 0.f ;  // ? 
}


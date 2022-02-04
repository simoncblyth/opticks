#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define TREE_FUNC __forceinline__ __device__
#else
#    define TREE_FUNC inline
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

#include "f4_stack.h"


/**
distance_tree : gets given trees with numNode 3, 7, 15, ... where some nodes can be CSG_ZERO empties
need to handle the tree without recursion, (the sdf approach of NNode.cpp in nunion::operator etc.. relies on recursion)


    0
  1   2


    0
  1    2
3  4  5  6


           0
     1           2
  3    4     5      6
7  8  9 10 11 12  13 14


Should be a lot simpler than intersect_tree as CSG union/intersection/difference can be 
done simply by fminf fmaxf on the distances from lower levels.  

Simply postorder traverse using a stack perhaps



**/


TREE_FUNC
float distance_tree( const float3& global_position, int numNode, const CSGNode* node, const float4* plan0, const qat4* itra0 )
{
    unsigned height = TREE_HEIGHT(numNode) ; // 1->0, 3->1, 7->2, 15->3, 31->4 

    // beginIdx, endIdx are 1-based level order tree indices, root:1 leftmost:1<<height    0:"parent" of root
    unsigned beginIdx = 1 << height ;  // leftmost 
    unsigned endIdx   = 0 ;            // parent of root 
    unsigned nodeIdx = beginIdx ; 

    F4_Stack stack ; 
    stack.curr = -1 ; 
    float distance = 0.f ; 

    while( nodeIdx != endIdx )
    {
        unsigned depth = TREE_DEPTH(nodeIdx) ;
        unsigned elevation = height - depth ; 

        const CSGNode* nd = node + nodeIdx - 1 ; 
        OpticksCSG_t typecode = (OpticksCSG_t)nd->typecode() ;

        if( typecode == CSG_ZERO )
        {
            nodeIdx = POSTORDER_NEXT( nodeIdx, elevation ) ;
            continue ; 
        }

        bool primitive = typecode >= CSG_SPHERE ;  
        // HMM : this includes CSG_CONTIGUOUS 
        if( primitive )
        {
            // could explicitly branch here on distance_node_contiguous or distance_leaf
            distance = distance_node(global_position, nd, plan0, itra0 ) ; 
            stack.push(distance) ; 
        }
        else
        {
            float lhs ;  
            float rhs ;  
            stack.pop2( lhs, rhs ); 

            switch( typecode )
            {
                case CSG_UNION:        distance = fminf( lhs,  rhs ) ; break ;   
                case CSG_INTERSECTION: distance = fmaxf( lhs,  rhs ) ; break ;   
                case CSG_DIFFERENCE:   distance = fmaxf( lhs, -rhs ) ; break ;   
                default:               distance = 0.f                ; break ;             
            }
            stack.push(distance) ; 
        }
        nodeIdx = POSTORDER_NEXT( nodeIdx, elevation ) ;
    }
    stack.pop(distance);  
    return distance ; 
}

/**
distance_list
--------------

Using fminf so is implicitly a union list, could also use fmaxf and have intersection list 
controlled by ancillary type on the head node. 

**/

TREE_FUNC
float distance_list( const float3& global_position, int numNode, const CSGNode* node, const float4* plan0, const qat4* itra0 )
{
    float distance = RT_DEFAULT_MAX ; 
    for(int nodeIdx=1 ; nodeIdx < numNode ; nodeIdx++)  // head node of list just carries subNum, so start from 1 
    {
        const CSGNode* nd = node + nodeIdx ; 
        float sd = distance_node(global_position, nd, plan0, itra0 ) ; 
        distance = fminf( distance,  sd ) ;   
    }
    return distance ; 
}



/**
intersect_tree
-----------------

http://xrt.wikidot.com/doc:csg

http://xrt.wdfiles.com/local--files/doc%3Acsg/CSG.pdf


                   
                  +-----------------------+
                  |                     B |
       +------------------+               | 
       | A        |       |               |
       |          |       |               |
       |          |       |               |
       |   0- - - 1 - - - 2 - - - - - - -[3]
       |          |       |               |
       |          |       |               |
       |          +-------|---------------+
       |                  |
       |                  |
       +------------------+

A ray is shot at each sub-object to find the nearest intersection, then the
intersection with the sub-object is classified as one of entering, exiting or
missing it. Based upon the combination of the two classifications, one of
several actions is taken:

1. returning a hit
2. returning a miss
3. changing the starting point of the ray for one of the objects and then
   shooting this ray, classifying the intersection. In this case, the state
   machine enters a new loop.

For full details of the LUT (lookup table) and the single-intersect CSG implementation below 
see tests/CSGClassifyTest.cc and run the below::

    CSGClassifyTest U
    CSGClassifyTest I
    CSGClassifyTest D


Complete binary tree of height 4 (31 nodes) with 1-based nodeIdx in binary:: 
                                                                                                                                          depth    elevation
                                                                      1                                                                      0         4
 
                                      10                                                            11                                       1         3

                          100                        101                            110                           [111]                       2         2

                 1000            1001          1010         1011             1100          1101            *1110*           1111              3         1
 
             10000  10001    10010 10011    10100 10101   10110 10111     11000 11001   11010  11011   *11100* *11101*   11110   11111          4         0
                                                                                                     


CSG looping in the below implementation has been using the below complete binary tree slices(tranche)::

    unsigned fullTree  = PACK4(  0,  0,  1 << height, 0 ) ;    

    unsigned leftIdx = 2*nodeIdx  ;    // left child of nodeIdx
    unsigned rightIdx = leftIdx + 1 ; // right child of nodeIdx  

    unsigned endTree   = PACK4(  0,  0,  nodeIdx,  endIdx  );
    unsigned leftTree  = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
    unsigned rightTree = PACK4(  0,  0,  rightIdx << (elevation-1), nodeIdx );


1 << height 
    leftmost, eg 10000
0 = 1 >> 1 
    one beyond root(1) in the sequence
 
nodeIdx
     node reached in the current slice of postorder sequence  
endIdx 
     one beyond the last node in the current sequence (for fulltree that is 0)

leftTree 
     consider example nodeIdx 111 which has elevation 2 in a height 4 tree
     
     nodeIdx  :  111
     leftIdx  : 1110  
     rightIdx : 1111

     leftTree.start : leftIdx << (2-1)  : 11100
     leftTree.end   : rightIdx << (2-1) : 11110    one beyond the leftIdx subtree of three nodes in the postorder sequence 

rightTree
    again consider nodeIdx 111

    nodeIdx   :  111
    rightIdx  : 1111

    rightTree.start : rightIdx << (2-1) : 11110     same one beyond end of leftTree is the start of the rightTree slice 
    rightTree.end   :nodeIdx 


Now consider how different things would be with an unbalanced tree : the number of nodes traversed in a leftTree traverse
of an unbalanced tree would be much more... the leftTree  would encompass the entirety of the postorder sequence 
up until the same end points as above.  The rightTree would not change.

Perhaps leftTreeOld should be replaced with leftTreeNew starting all the way from leftmost beginning of the postorder sequence::

    unsigned leftTreeOld  = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
    unsigned leftTreeNew  = PACK4(  0,  0,  1 << height , rightIdx << (elevation-1)) ; 

I suspect that when using balanced trees the leftTreeold can cause spurious intersects
due to discontiguity from incomplete geometry as a result of not looping over the 
full prior postorder sequence. 

Tried using leftTreeNew with a balanced tree and it still gives spurious intersects on internal boundariues,
so it looks like tree balanching and the CSG algorithm as it stands are not compatible.  

TODO: need to confirm exactly what is happening using CSGRecord, have not extinguished all hope of getting balanced
to work yet, as it seems like it should be possible in principle.

**/

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

#ifdef DEBUG_RECORD
        if(CSGRecord::ENABLED)
            printf("// tranche_pop : tloop %d tmin %10.4f beginIdx %d endIdx %d  tr.curr %d csg.curr %d   \n", tloop, tmin, beginIdx, endIdx, tr.curr, csg.curr );
        assert( ierr == 0 ); 
#endif

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


#ifdef DEBUG_RECORD
                if(CSGRecord::ENABLED)
                {
                    IntersectionState_t nd_state = CSG_CLASSIFY( nd_isect,  ray_direction, tmin );
                    printf("// %3d : primative push : add record  \n", nodeIdx ); 
                    quad4& rec = CSGRecord::record.back(); 
                    rec.q1.f.x = ray_origin.x + fabs(rec.q0.f.w)*ray_direction.x ; 
                    rec.q1.f.y = ray_origin.y + fabs(rec.q0.f.w)*ray_direction.y ; 
                    rec.q1.f.z = ray_origin.z + fabs(rec.q0.f.w)*ray_direction.z ; 

                    rec.q2.i.x = typecode ; 
                    rec.q2.i.y = int(nd_state) ; 
                    rec.q2.i.z = -1 ; 
                    rec.q2.i.w = -1 ; 

                    rec.q3.i.x = tloop ; 
                    rec.q3.i.y = nodeIdx ;    // mark primitive push 
                    rec.q3.i.z = -1 ;         // placeholder for ctrl 
                  
                }
                assert( ierr == 0 ); 
#endif
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

                // operator node : peek at the top of the stack 

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


                // it is impossible to Miss a complemented (signaled by -0.f) solid as it is unbounded
                // hence the below artificially changes leftIsCloser to reflect the unboundedness 
                // and sets the corresponding states to Exit
                // see opticks/notes/issues/csg_complement.rst 
                // these settings are only valid (and only needed) for misses 

                bool l_complement = signbit(csg.data[left].x) ;
                bool r_complement = signbit(csg.data[right].x) ;

                bool l_complement_miss = l_state == State_Miss && l_complement ;
                bool r_complement_miss = r_state == State_Miss && r_complement ;

                if(r_complement_miss)
                {
#ifdef DEBUG_RECORD
                    if(CSGRecord::ENABLED)
                    {
                        printf("// %3d : r_complement_miss setting leftIsCloser %d to true and r_state %5s to Exit \n", 
                                nodeIdx, leftIsCloser, IntersectionState::Name(l_state)  ); 
                    }
#endif
                    r_state = State_Exit ; 
                    leftIsCloser = true ; 
               } 

                if(l_complement_miss)
                {
#ifdef DEBUG_RECORD
                    if(CSGRecord::ENABLED)
                    {
                        printf("// %3d : l_complement_miss setting leftIsCloser %d to false and l_state %5s to Exit \n", 
                                nodeIdx, leftIsCloser, IntersectionState::Name(r_state)  ); 
                    }
#endif
                    l_state = State_Exit ; 
                    leftIsCloser = false ; 
                } 

                int ctrl = lut.lookup( typecode , l_state, r_state, leftIsCloser ) ;

#ifdef DEBUG_RECORD
                if(CSGRecord::ENABLED)
                {
                    printf("// %3d : stack peeking : left %d right %d (stackIdx)  %15s  l:%5s %10.4f    r:%5s %10.4f     leftIsCloser %d -> %s \n", 
                           nodeIdx,left,right,
                           CSG::Name(typecode), 
                           IntersectionState::Name(l_state), t_left,  
                           IntersectionState::Name(r_state), t_right, 
                           leftIsCloser, 
                           CTRL::Name(ctrl)  ); 
                }
#endif

                Action_t act = UNDEFINED ; 

                if(ctrl < CTRL_LOOP_A) // non-looping : CTRL_RETURN_MISS/CTRL_RETURN_A/CTRL_RETURN_B/CTRL_RETURN_FLIP_B "returning" with a push 
                {
                    float4 result = ctrl == CTRL_RETURN_MISS ?  make_float4(0.f, 0.f, 0.f, 0.f ) : csg.data[ctrl == CTRL_RETURN_A ? left : right] ;
                    if(ctrl == CTRL_RETURN_FLIP_B)
                    {
                        result.x = -result.x ;     
                        result.y = -result.y ;     
                        result.z = -result.z ;     
                    }
                    result.w = copysignf( result.w , nodeIdx % 2 == 0 ? -1.f : 1.f );  
                    // record left/right in sign of t 

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

                    unsigned endTree       = PACK4(  0,  0,  nodeIdx,  endIdx  );
                    unsigned leftTree      = PACK4(  0,  0,  leftIdx << (elevation-1), rightIdx << (elevation-1)) ;
                    //unsigned leftTreeNew   = PACK4(  0,  0,  1 << height             , rightIdx << (elevation-1)) ; 
                    unsigned rightTree     = PACK4(  0,  0,  rightIdx << (elevation-1), nodeIdx );

                    unsigned loopTree  = ctrl == CTRL_LOOP_A ? leftTree : rightTree  ;

#ifdef DEBUG_RECORD
                    if(CSGRecord::ENABLED) 
                    printf("// %3d : looping one side tminAdvanced %10.4f with eps %10.4f \n", nodeIdx, tminAdvanced, propagate_epsilon );  
#endif


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

                   // push the tranche from here to endTree before pushing the backtracking tranche so known how to proceed after backtracking done
                   // (hmm: using tmin onwards to endTree looks a bit funny, maybe it should be advanced?)

                    ierr = tranche_push( tr, endTree,  tmin );         if(ierr) break ;   
                    ierr = tranche_push( tr, loopTree, tminAdvanced ); if(ierr) break ; 

                    act = BREAK  ;  

#ifdef DEBUG_RECORD
                    if(CSGRecord::ENABLED) 
                    printf("// %3d : looping :  act BREAK \n", nodeIdx ); 
#endif

                }                      // "return" or "recursive call" 

#ifdef DEBUG_RECORD
                if(CSGRecord::ENABLED) 
                {
                    //printf("// %3d : add record \n", nodeIdx ); 
                    quad4& rec = CSGRecord::record.back(); 

                    rec.q1.f.x = ray_origin.x + fabs(rec.q0.f.w)*ray_direction.x ; 
                    rec.q1.f.y = ray_origin.y + fabs(rec.q0.f.w)*ray_direction.y ; 
                    rec.q1.f.z = ray_origin.z + fabs(rec.q0.f.w)*ray_direction.z ; 

                    rec.q2.i.x = typecode ; //  set to initial csg.curr within csg_push  
                    rec.q2.i.y = int(l_state) ; 
                    rec.q2.i.z = int(r_state) ; 
                    rec.q2.i.w = int(leftIsCloser) ; 

                    rec.q3.i.x = tloop ;  
                    rec.q3.i.y = nodeIdx ;    // mark operator push with negation 
                    rec.q3.i.z = ctrl  ; 
                }
#endif

                if(act == BREAK) 
                {
#ifdef DEBUG_RECORD
                     if(CSGRecord::ENABLED) 
                     printf("// %3d : break for backtracking \n", nodeIdx ); 
#endif
                     break ; 
                }
            }                          // "primitive" or "operation"
            nodeIdx = POSTORDER_NEXT( nodeIdx, elevation ) ;
        }                     // node traversal 
        if(ierr) break ; 
    }                        // subtree tranches

    ierr |= (( csg.curr !=  0)  ? ERROR_END_EMPTY : 0)  ; 

#ifdef DEBUG_RECORD
    if(CSGRecord::ENABLED) 
    printf("// intersect_tree.h ierr %d csg.curr %d \n", ierr, csg.curr ); 
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





TREE_FUNC
bool intersect_prim( float4& isect, int numNode, const CSGNode* node, const float4* plan, const qat4* itra, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
#ifdef DEBUG6
#if OPTIX_VERSION < 70000 
    rtPrintf("// intersect_prim  numNode %d \n", numNode );  
#endif
#endif

    return numNode == 1 
               ? 
                  intersect_node(isect,          node, plan, itra, t_min, ray_origin, ray_direction )
               :
                  //intersect_tree(isect, numNode, node, plan, itra, t_min, ray_origin, ray_direction )
                  intersect_node_contiguous( isect, node, plan, itra, t_min, ray_origin, ray_direction )
               ;
}


TREE_FUNC
float distance_prim( const float3& global_position, int numNode, const CSGNode* node, const float4* plan, const qat4* itra )
{
    return numNode == 1 
               ? 
                  distance_node(global_position,          node, plan, itra )
               :
                  distance_tree(global_position, numNode, node, plan, itra )
               ;
}



#pragma once

#ifdef __CUDACC__
#define POSTORDER_DEPTH(currIdx) ( 32 - __clz((currIdx)) - 1 )
#else
#define POSTORDER_DEPTH(currIdx) ( 32 - std::__clz((currIdx)) - 1 )
// suspect above is clang specific, TODO find gcc equiv and detect for it 
#endif


// see dev/csg/postorder.py 
#define POSTORDER_NEXT(currIdx, elevation )( ((currIdx) & 1) ? (currIdx) >> 1 :  ((currIdx) << (elevation)) + (1 << (elevation)) )


// perfect binary tree assumptions,   2^(h+1) - 1 
#define TREE_HEIGHT(numNodes) ( __ffs((numNodes) + 1) - 2)
#define TREE_NODES(height) ( (0x1 << (1+(height))) - 1 )
#define TREE_DEPTH(nodeIdx) ( 32 - __clz((nodeIdx)) - 1 )





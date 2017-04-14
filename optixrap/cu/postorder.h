#pragma once

#ifdef __CUDACC__
#define POSTORDER_DEPTH(currIdx) ( 32 - __clz((currIdx)) - 1 )
#else
#define POSTORDER_DEPTH(currIdx) ( 32 - std::__clz((currIdx)) - 1 )
// suspect above is clang specific, TODO find gcc equiv and detect for it 
#endif


// see dev/csg/postorder.py 
#define POSTORDER_NEXT(currIdx, elevation )( ((currIdx) & 1) ? (currIdx) >> 1 :  ((currIdx) << (elevation)) + (1 << (elevation)) )




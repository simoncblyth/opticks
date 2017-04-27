#pragma once
/*

1-based indexing of complete binary tree
-------------------------------------------

Exhibits a very regular pattern of the bits::

                                                     depth     elevation

                         1                               0           3   

              10                   11                    1           2   

         100       101        110        111             2           1   
                        
     1000 1001  1010 1011  1100 1101  1110  1111         3           0   
 

This has several advantages:

* child/parent indices can be computed (not stored) so no tree overheads
* postorder traverse can be computed by bit twiddling 

::

    parent(i)         = i/2 
    leftchild(i)      = 2*i + 1 
    rightchild(i)     = 2*i + 2 

    leftmost(height)  =  1 << height 
    postorder_next(i) =  i & 1   ?   i >> 1   :   (i << elevation) + (1 << elevation)  


*/

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





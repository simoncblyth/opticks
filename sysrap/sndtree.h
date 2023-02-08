#pragma once
/**
sndtree.h 
===========

Need some of the functionality of the old NTreeBuilder for use with snd node trees 


See tests/sndtree_test.cc

snd::render width 7 height 2
            .                   
                                
    .               .           
                                
.       .       .       .       
                                

 snd::Inorder [ 4 6 5 10 7 9 8 ]

Notice that the node indices here all exceed the spheres 0,1,2,3 
as the operator nodes and CSG_ZERO nodes are all created afterwards. 

Perhaps can avoid wasting node indices by directly popping the leaves
and using them during the build ? 

* Is that too constraining ? 
* What about pruning ? 

The snd approach of auto-adding snd to the POOL
has problem of wasting nodes once things get complicated. 

Its kinda like I need to build the tree in a separate scsg sandbox 
and then only bring it over into the standard POOL when all the dirty stuff 
like pruning etc.. is done.  

But that means have two "address spaces" for node indices ? 
How could that work ?

Maybe can avoid that by setting the POOL to the SANDBOX scsg
prior to creating the leaves even : so can do everything 
with wild abandon about wasting nodes in the SANDBOX. 

snd is using integers as pointers to nodes (because that 
can be persisted) but unlike pointers which can be deleted 
there is currently no way to delete the integer-pointer-nodes.  

HMM BUT there could be ? That would mean updating integer refs elsewhere ?


Note that the kinda nodes that are needed for tree creation and pruning  
are very generic and simple (boolean is enough). So maybe just use  
simple pointer based nd to hack out the tree. 

**/

#include <cassert>
#include <vector>
#include "snd.hh"

struct sndtree
{
    static int FindBinaryTreeHeight(int num_leaves); 
    static int CommonTree( const std::vector<int>& nodes, int op ) ; 
    static int Build_r( int elevation, int op ); 
}; 

/**
sndtree::FindBinaryTreeHeight
-------------------------------

Return complete binary tree height sufficient for num_leaves
        
   height: 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10, 
   tprim : 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 


                          1                                  h=0,  1    

            10                        11                     h=1,  2 

      100         101          110            111            h=2,  4 

   1000 1001  1010  1011   1100   1101     1110  1111        h=3,  8
        

**/


int sndtree::FindBinaryTreeHeight(int num_leaves) // static
{
    int  height = -1 ;
    for(int h=0 ; h < 10 ; h++ )
    {   
        int tprim = 1 << h ;   
        if( tprim >= num_leaves )
        {   
           height = h ; 
           break ;
        }   
    }   
    assert(height > -1 );  
    return height ; 
}

/**
sndtree::CommonTree
---------------------

Start with a simple subset of NTreeBuilder hoping can avoid the more involved stuff

**/

inline int sndtree::CommonTree( const std::vector<int>& nodes, int op ) // static
{
    int num_leaves = nodes.size() ; 
    int height = FindBinaryTreeHeight(num_leaves) ; 
    int root = Build_r( height, op );  
    return root ; 
}

/**
sndtree::Build_r
------------------

**/

inline int sndtree::Build_r( int elevation, int op )
{
    int l = elevation > 1 ? Build_r( elevation - 1 , op ) : snd::Zero() ; 
    int r = elevation > 1 ? Build_r( elevation - 1 , op ) : snd::Zero() ; 
    return snd::Boolean( op, l, r ); 
}


#pragma once
/**
snd_TreeBuilder.h
==================

HMM: unlike npy/NTreeBuilder the snd.hh node tree 
is not limited to being binary 

So maybe no need for snd_TreeBuilder, can handle it 
like a multiunion ? 

This of course just defers the issue to CSGNode time.
But its the right thing to do. 

**/

#include <string>
#include <sstream>
#include <vector>
#include "snd.hh"

struct snd_TreeBuilder
{
    std::string desc() const ; 
    static int BinaryTreeHeight(unsigned num_leaves);  
    snd_TreeBuilder(const std::vector<int>& prims ); 

    const std::vector<int>& prims ; 
    int height ; 
}; 

std::string snd_TreeBuilder::desc() const 
{
    std::stringstream ss ; 
    ss << "snd_TreeBuilder::desc" 
       << " prims " << prims.size()
       << " height " << height 
       ;
    std::string str = ss.str(); 
    return str ; 

}



/**
snd_TreeBuilder::BinaryTreeHeight
---------------------------------------

Return complete binary tree height sufficient for num_leaves
        
   height: 0, 1, 2, 3,  4,  5,  6,   7,   8,   9,   10, 
   tprim : 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 

**/
inline int snd_TreeBuilder::BinaryTreeHeight(unsigned num_leaves) // static
{
    int h = -1 ;
    for(int i=0 ; i < 10 ; i++ )
    {   
        int tprim = 1 << i ;   
        if( tprim >= int(num_leaves) )
        {   
           h = i ; 
           break ;
        }   
    }   
    assert(h > -1 );  
    return h ; 
}

inline snd_TreeBuilder::snd_TreeBuilder(const std::vector<int>& prims_ )
    :
    prims(prims_),
    height(BinaryTreeHeight(prims.size()))
{
}


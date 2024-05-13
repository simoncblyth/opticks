#pragma once
/**
st : complete binary tree expressions 
========================================

cf with CSG/csg_postorder.h that is used on GPU for these 


// tree_nodes_ = lambda height:( (0x1 << (1+(height))) - 1 ) 

**/

#include <cstdint>

struct st
{
    static constexpr const uint64_t ONE = 1 ;  
    static uint64_t complete_binary_tree_nodes(uint64_t height) ; 
};

inline uint64_t st::complete_binary_tree_nodes(uint64_t height)
{
    return ( (ONE << (ONE+(height))) - ONE ) ; 
}



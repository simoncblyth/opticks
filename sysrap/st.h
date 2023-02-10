#pragma once
/**
st : complete binary tree expressions 
========================================

cf with CSG/csg_postorder.h that is used on GPU for these 


// tree_nodes_ = lambda height:( (0x1 << (1+(height))) - 1 ) 

**/

struct st
{
    static int complete_binary_tree_nodes(int height) ; 
};

inline int st::complete_binary_tree_nodes(int height)
{
    return ( (0x1 << (1+(height))) - 1 ) ; 
}



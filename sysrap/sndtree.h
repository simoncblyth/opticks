#pragma once
/**
sndtree.h
===========

Used from snd::UnionTree which is for example used by U4Polycone 
This brings some of the functionality of the old NTreeBuilder for use with snd node trees 

This is a workaround combining the inherently persistable but not very flexible
snd.hh with the more flexible pointer based sn.h 

Once sn.h persisting using s_pool.h becomes fully featued 
this sndtree.h can be removed. 

**/

#include <cassert>
#include <vector>
#include "snd.hh"
#include "sn.h"

struct sndtree
{
    static int CommonTree( const std::vector<int>& nodes, int op ) ; 
    static int Build_r(sn* n, int& num_leaves_placed, const std::vector<int>& leaves, int d ); 
}; 


/**
sndtree::CommonTree
---------------------

Creates *snd* binary tree sufficient to hold the 
leaves and places the leaf references into the tree.  

Internally this uses the pointer based *sn* tree as a guide, 
assisting with the dirty work of creating the complete binary tree 
and then pruning it down to hold the leaves provided. 
The *sn* nodes are used for this as they can easily be deleted (or leaked)
unlike the *snd* nodes. 

Constructing the *snd* tree requires calling snd::Boolean with 3 int args
in a postorder traverse to cover children before parents.  
The snd::Boolean call does the n-ary setup for the 2-ary boolean nodes.

**/

inline int sndtree::CommonTree( const std::vector<int>& leaves, int op ) // static
{
    int num_leaves = leaves.size() ; 
    std::vector<int> leaftypes ; 
    snd::GetTypes(leaftypes, leaves); 

    sn* n = sn::CommonTree( leaftypes, op ); 

    int num_leaves_placed = 0 ; 
    int root = Build_r(n, num_leaves_placed, leaves, 0 );  
    assert( num_leaves_placed == num_leaves );  

    return root ; 
}


/**
sndtree::Build_r
------------------

Postorder visit after recursive call : so children reached before parents  

**/

inline int sndtree::Build_r(sn* n, int& num_leaves_placed, const std::vector<int>& leaves, int d )
{
    int N = -1 ; 
    if( n->is_operator() )
    {
        int op = n->type ; 
        int L = Build_r(n->left,  num_leaves_placed, leaves, d+1) ; 
        int R = Build_r(n->right, num_leaves_placed, leaves, d+1) ; 
        N = snd::Boolean( op, L, R );  
    }
    else
    {
        N = leaves[num_leaves_placed] ; 
        num_leaves_placed += 1 ; 
    }
    return N ; 
}



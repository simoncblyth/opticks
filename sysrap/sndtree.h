#pragma once
/**
sndtree.h
===========

Used from snd::UnionTree which is for example used by U4Polycone 
This brings some of the functionality of the old NTreeBuilder for use with snd node trees 

This is a workaround combining the inherently persistable but not very flexible
snd.hh with the more flexible pointer based sn.h 

Once sn.h persisting using s_pool.h s_csg.h becomes fully featured 
this sndtree.h SHOULD BE REMOVED. 
 
AS ITS AN UNHEALTHY MIX OF TWO NODE TYPES THAT ARE DOING 
ESSENTIALLY THE SAME THING 

**/

#include <cassert>
#include <vector>
#include "snd.hh"
#include "sn.h"

struct sndtree
{
    static int CommonTree_PlaceLeaves( const std::vector<int>& leaves, int op ) ; 
    static int Build_r(sn* n, int& num_leaves_placed, const std::vector<int>& leaves, int d ); 
}; 


/**
sndtree::CommonTree_PlaceLeaves
---------------------------------

The *leaves* are snd indices, so can snd::Get(idx) to access the snd 

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

inline int sndtree::CommonTree_PlaceLeaves( const std::vector<int>& leaves, int op ) // static
{
    int num_leaves = leaves.size() ; 
    std::vector<int> leaftypes ; 
    snd::GetTypes(leaftypes, leaves); 

    sn* n = sn::CommonOperatorTypeTree( leaftypes, op ); 

    int num_leaves_placed = 0 ; 
    int root = Build_r(n, num_leaves_placed, leaves, 0 );  

    bool expect_leaves = num_leaves_placed == num_leaves ;
    if(!expect_leaves) std::cerr << "sndtree::CommonTree_PlaceLeaves UNEXPECTED LEAVES " << std::endl; 
    assert( expect_leaves );  

    delete n ; 

    snd* r = snd::Get_(root); 
    r->sibdex = 0 ; 

    return root ; 
}


/**
sndtree::Build_r
------------------

Builds snd tree based on the "skeleton" provided by the sn tree.

Postorder visit after recursive call : so children reached before parents  

**/

inline int sndtree::Build_r(sn* n, int& num_leaves_placed, const std::vector<int>& leaves, int d )
{
    int N = -1 ; 
    if( n->is_operator() )
    {
        int op = n->typecode ; 
        int nc = n->num_child();  
        bool nc_expect = nc == 2 ; 
        if(!nc_expect) std::cerr << "sndtree::Build_r nc_expect " << std::endl ; 
        assert( nc_expect ); 
        sn* l = n->get_child(0); 
        sn* r = n->get_child(1); 
        int L = Build_r(l, num_leaves_placed, leaves, d+1) ; 
        int R = Build_r(r, num_leaves_placed, leaves, d+1) ; 
        N = snd::Boolean( op, L, R );  
    }
    else
    {
        N = leaves[num_leaves_placed] ; 
        num_leaves_placed += 1 ; 
    }
    return N ; 
}



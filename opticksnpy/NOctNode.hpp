#pragma once

/*

NOctNode
===========

* https://github.com/nickgildea/DualContouringSample
* http://ngildea.blogspot.tw/2014/11/implementing-dual-contouring.html
* http://www.frankpetterson.com/publications/dualcontour/dualcontour.pdf

Started out with aim to reimplement DualContouring, 
but it turns out to be too involved. So aborted that
and instead just bring it in wholesale, as a temporary 
measure... to see the meshes it produces before putting 
much time into it.







From Ericson p309

* complete binary tree of n levels has 2^n − 1 nodes
* complete d-ary tree of n levels has (d^n − 1)/(d − 1) nodes.
* octree implicit indices: 8*i + 1, ... 8*i + 8 like binary trees, 

Not so useful as complete octrees extrememly wasteful 

*/

#include <string>
#include "NQuad.hpp"
#include "NPY_API_EXPORT.hh"

struct NPY_API NLeafData
{
   int corners ; 
};

#include "NOctNodeEnum.hpp"

struct NPY_API NOctNode
{
    static const nivec3 OFFSETS[8] ;
     
    static float sdf( const nivec3& min, const float scale);
    static int   sdf_corners( const nivec3& min, const int size, const float scale );

    static NOctNode* Construct(const nivec3& min, const int size, const float scale);
    static NOctNode* Construct_r( NOctNode* parent);
    static NOctNode* ConstructLeaf(     NOctNode* node);
    static void Traverse(NOctNode* node, int depth=0);
    static int TraverseIt(NOctNode* node);

    NOctNode( NOctNode_t type );

    std::string desc();

    NOctNode_t type ; 
    NOctNode*  child[8] ; 
    nivec3     min ;  
    int        size ;     
    float      scale ; 
    NLeafData* data ; 
};






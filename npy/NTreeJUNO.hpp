#pragma once

#include <string>
#include <vector>
#include "NPY_API_EXPORT.hh"
struct nnode ; 
struct ncone ; 

/**
NTreeJUNO
===========

Rationalization:

1. replaces (tubs-torus) with cone 
2. replaces vacuum cap intersect/subtract with ellipsoid z-cuts
   (in the case of LV 18 this simplifies the tree to a single primitive 
    z-cut ellipsoid)   

**/

struct NPY_API NTreeJUNO
{
    static nnode* Rationalize(nnode* a);

    NTreeJUNO(nnode* root_) ;
    nnode* root ; 
    ncone* cone ; 

    ncone* replacement_cone() const ; 
    void rationalize();

    static nnode* create(int lv);  
    // lv:18,19,20,21  negated lv are rationalized 

    typedef std::vector<int> VI ;
    static const VI LVS ; 


};



#pragma once

#include <vector>

struct nnode ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NNodeSample 
{
    static void Tests(std::vector<nnode*>& nodes );

    static nnode* Sample(const char* name);
    static nnode* DifferenceOfSpheres(); 
    static nnode* Box3(); 

    static nnode* _Prepare(nnode* root); 
};



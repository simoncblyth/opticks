#pragma once

#include <vector>

struct nnode ; 

#include "NPY_API_EXPORT.hh"

struct NPY_API NNodeSample 
{
    static void Tests(std::vector<nnode*>& nodes );
    static nnode* Sphere1(); 
    static nnode* Sphere2(); 
    static nnode* Union1(); 
    static nnode* Intersection1(); 
    static nnode* Difference1(); 
    static nnode* Difference2(); 
    static nnode* Union2(); 
    static nnode* Box(); 
    static nnode* SphereBoxUnion(); 
    static nnode* SphereBoxIntersection(); 
    static nnode* SphereBoxDifference(); 
    static nnode* BoxSphereDifference(); 

    static nnode* Sample(const char* name);
    static nnode* DifferenceOfSpheres(); 
    static nnode* Box3(); 

    static nnode* _Prepare(nnode* root); 
};



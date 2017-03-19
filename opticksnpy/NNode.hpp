#pragma once

#include "OpticksCSG.h"
#include "NPY_API_EXPORT.hh"

struct nbbox ; 

struct NPY_API nnode {
    virtual double operator()(double px, double py, double pz) ;
    virtual void dump(const char* msg="nnode::dump");
    virtual const char* csgname(); 
    virtual nbbox bbox();

    OpticksCSG_t type ;  
    nnode* left ; 
    nnode* right ; 
};

// hmm perhaps easier to switch on these ?? instead
// of having separate types ? 

struct NPY_API nunion : nnode {
    double operator()(double px, double py, double pz) ;
};

inline NPY_API nunion make_nunion(nnode* left, nnode* right)
{
    nunion u ; u.type = CSG_UNION ; u.left = left ; u.right = right ; return u ; 
}


struct NPY_API nintersection : nnode {
    double operator()(double px, double py, double pz);
};

inline NPY_API nintersection make_nintersection(nnode* left, nnode* right)
{
    nintersection i ; i.type = CSG_INTERSECTION ; i.left = left ; i.right = right ; return i ; 
}


struct NPY_API ndifference : nnode {
    double operator()(double px, double py, double pz);
};
inline NPY_API ndifference make_ndifference(nnode* left, nnode* right)
{
    ndifference d ; d.type = CSG_DIFFERENCE ; d.left = left ; d.right = right ; return d ; 
}




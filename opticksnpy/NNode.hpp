#pragma once

#include "OpticksCSG.h"
#include "NPY_API_EXPORT.hh"

struct nbbox ; 

struct NPY_API nnode {
    virtual double operator()(double px, double py, double pz) ;
    virtual void dump(const char* msg="nnode::dump");
    virtual const char* csgname(); 
    virtual nbbox bbox();
    virtual unsigned maxdepth();
    virtual unsigned _maxdepth(unsigned depth);

    static void Init(nnode& n, OpticksCSG_t type, nnode* left=NULL, nnode* right=NULL);

    OpticksCSG_t type ;  
    nnode* left ; 
    nnode* right ; 
};

// hmm perhaps easier to switch on these ?? instead
// of having separate types ? 

struct NPY_API nunion : nnode {
    double operator()(double px, double py, double pz) ;
};
struct NPY_API nintersection : nnode {
    double operator()(double px, double py, double pz);
};
struct NPY_API ndifference : nnode {
    double operator()(double px, double py, double pz);
};


inline NPY_API nunion make_nunion(nnode* left, nnode* right)
{
    nunion n ;         nnode::Init(n, CSG_UNION , left, right ); return n ; 
}
inline NPY_API nintersection make_nintersection(nnode* left, nnode* right)
{
    nintersection n ;  nnode::Init(n, CSG_INTERSECTION , left, right ); return n ;
}
inline NPY_API ndifference make_ndifference(nnode* left, nnode* right)
{
    ndifference n ;    nnode::Init(n, CSG_DIFFERENCE , left, right ); return n ;
}

inline NPY_API nunion* make_nunion_ptr(nnode* left, nnode* right)
{
    nunion* n = new nunion ;         nnode::Init(*n, CSG_UNION , left, right ); return n ; 
}
inline NPY_API nintersection* make_nintersection_ptr(nnode* left, nnode* right)
{
    nintersection* n = new nintersection ;  nnode::Init(*n, CSG_INTERSECTION , left, right ); return n ;
}
inline NPY_API ndifference* make_ndifference_ptr(nnode* left, nnode* right)
{
    ndifference* n = new ndifference ;    nnode::Init(*n, CSG_DIFFERENCE , left, right ); return n ;
}




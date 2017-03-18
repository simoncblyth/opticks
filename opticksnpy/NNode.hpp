#pragma once

#include "OpticksCSG.h"
#include "NPY_API_EXPORT.hh"

struct NPY_API nnode {
    virtual double operator()(double px, double py, double pz) ;
    virtual void dump(const char* msg="nnode::dump");
    const char* csgname(); 
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



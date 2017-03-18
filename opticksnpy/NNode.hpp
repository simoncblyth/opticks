#pragma once

#include "NPY_API_EXPORT.hh"

struct NPY_API nnode {
   virtual double operator()(double px, double py, double pz) ;
};

struct NPY_API nunion : nnode {
    double operator()(double px, double py, double pz) ;
    nnode* left ; 
    nnode* right ; 
};
struct NPY_API nintersection : nnode {
    double operator()(double px, double py, double pz);
    nnode* left ; 
    nnode* right ; 
};
struct NPY_API ndifference : nnode {
    double operator()(double px, double py, double pz);
    nnode* left ; 
    nnode* right ; 
};




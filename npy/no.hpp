#pragma once

#include "NPY_API_EXPORT.hh"
#include "OpticksCSG.h"
#include <string>

// minimal node class standin for nnode used to develop tree machinery 

struct NPY_API no
{
    const char* label ; 
    no* left ; 
    no* right ; 
    unsigned depth ;
    OpticksCSG_t type ; 
 
    std::string desc() const ;
    bool is_primitive() const ; 
    bool is_operator() const ; 
    bool is_zero() const ; 

    bool is_lrzero() const ;  //  l-zero AND  r-zero
    bool is_rzero() const ;   // !l-zero AND  r-zero
    bool is_lzero() const ;   //  l-zero AND !r-zero

    static no make_node( OpticksCSG_t type, no* left=NULL, no* right=NULL );  

}; 



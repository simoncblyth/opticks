#pragma once

struct nmat4triple ;
template <typename T> class NPY ; 

#include "NPY_API_EXPORT.hh"
/**
nxform
======

Thought this might enable avoiding duplication 
of transform mechanics.  But the users of this
are nodes within a node tree, so its kinda complex
to make this available.


**/


struct NPY_API nxform
{
    nxform(unsigned num_nodes_, bool debug_);

    const nmat4triple* make_triple( const float* data);
    
    template <typename N>
    static const nmat4triple* make_global_transform(const N* n)  ;

    unsigned     num_nodes ; 
    bool         debug ; 
    unsigned     num_triple  ; 
    unsigned     num_triple_mismatch ;
    NPY<float>*  triple ;
  
};

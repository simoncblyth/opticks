#pragma once

#include "NNode.hpp"
#include "NPY_API_EXPORT.hh"
struct nbbox ; 

/*

nmultiunion
=============

*/

struct nmat4triple ; 

struct NPY_API nmultiunion : nnode 
{
    nbbox bbox() const ; 
    float operator()(float x_, float y_, float z_) const ; 


    // placeholder zeros
    int par_euler() const ;
    unsigned par_nsurf() const ;
    unsigned par_nvertices(unsigned , unsigned ) const ; 

};


inline NPY_API nmultiunion* make_multiunion(OpticksCSG_t type )
{
    nmultiunion* n = new nmultiunion ; 
    //assert( type == CSG_CONTIGUOUS || type == CSG_DISCONTIGUOUS); 
    assert( type == CSG_CONTIGUOUS  ); 
    nnode::Init(n,type) ; 

    return n ; 
}
inline NPY_API nmultiunion* make_multiunion(OpticksCSG_t type, const nquad& param  )
{
    nmultiunion* n = make_multiunion(type) ; 
    n->param = param ;    
    return n ; 
}

inline NPY_API nmultiunion* make_multiunion(OpticksCSG_t type, unsigned sub_num  )
{
    nmultiunion* n = make_multiunion(type) ; 
    n->setSubNum(sub_num); 
    return n ; 
}



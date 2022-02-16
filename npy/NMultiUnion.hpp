#pragma once

#include <vector>
#include "plog/Severity.h"

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
    static const plog::Severity LEVEL ; 

    static nmultiunion* Create(OpticksCSG_t type) ; 
    static nmultiunion* Create(OpticksCSG_t type, const nquad& param  ); 
    static nmultiunion* Create(OpticksCSG_t type, unsigned sub_num  ); 

    static nmultiunion* CreateFromTree( OpticksCSG_t type, const nnode* subtree ); 
    static nmultiunion* CreateFromList( OpticksCSG_t type, std::vector<nnode*>& prim  ); 

    nbbox bbox() const ; 
    float operator()(float x_, float y_, float z_) const ; 


    // placeholder zeros
    int par_euler() const ;
    unsigned par_nsurf() const ;
    unsigned par_nvertices(unsigned , unsigned ) const ; 

};



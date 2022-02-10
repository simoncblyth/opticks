#pragma once

#include "NNode.hpp"
#include "plog/Severity.h"
#include "NPY_API_EXPORT.hh"
struct nbbox ; 

/*

nphicut
=============

*/

struct nmat4triple ; 

struct NPY_API nphicut : nnode 
{
    static const plog::Severity LEVEL ; 
    static nphicut* make(OpticksCSG_t type, double startPhi_pi, double deltaPhi_pi );

    void pdump(const char* msg) const ; 

    float operator()(float x_, float y_, float z_) const ; 

    glm::vec3 normal(int idx) const ; 

    // placeholder : otherwise X4PhysicalVolume::ConvertSolid_FromRawNode asserts in NCSG::Adopt nnode::collectParPoints
    int par_euler() const ;
    unsigned par_nsurf() const ;
    unsigned par_nvertices(unsigned , unsigned ) const ; 

};



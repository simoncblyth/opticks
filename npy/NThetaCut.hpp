#pragma once

#include "NNode.hpp"
#include "plog/Severity.h"

#include "NPY_API_EXPORT.hh"
struct nbbox ; 

/*

nthetacut
=============

*/

struct nmat4triple ; 

struct NPY_API nthetacut : nnode 
{
    static const plog::Severity LEVEL ; 

    static nthetacut* make(double startTheta_pi, double deltaTheta_pi); 

    float operator()(float x_, float y_, float z_) const ; 
    void pdump(const char* msg) const ; 

    // placeholders : without then get NCSG::Adopt assert with X4PhysicalVolume::ConvertSolid_FromRawNode
    int par_euler() const ;
    unsigned par_nsurf() const ;
    unsigned par_nvertices(unsigned , unsigned ) const ; 

};



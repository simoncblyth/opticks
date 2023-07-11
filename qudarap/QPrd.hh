#pragma once
/**
QPrd : used by QSimTest/mock_propagate
=========================================

Dummy per-ray-data enabling pure-CUDA (no OptiX, no geometry) 
testing of propagation using QSimTest MOCK_PROPAGATE.

MUST be instanciated after QBnd 

The basis vectors of are obtained
from envvars QPRD_BND and QPRD_NRMT
which have defaults. 

    boundary_idx
    nrmt:(normals,distance) 




**/

#include "QUDARAP_API_EXPORT.hh"
#include <vector>
#include <string>
#include "plog/Severity.h"

struct float4 ; 
struct quad2 ; 
struct QBnd ; 
struct SBnd ; 
struct NP ; 

struct QUDARAP_API QPrd
{
    static const plog::Severity LEVEL ;
    static const QPrd* INSTANCE  ; 
    static const QPrd* Get() ; 

    const QBnd* bnd ; 
    const SBnd* sbn ; 

    static constexpr const char* QPRD_NRMT_DEFAULT = "0,0,1,100 0,0,1,200 0,0,1,300 0,0,1,400" ; 
    static constexpr const char* QPRD_BND_DEFAULT = R"LITERAL(
    Acrylic///LS
    Water///Acrylic
    Water///Pyrex
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    )LITERAL" ;  

    // bnd name changed 
    // Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum


    std::vector<int>    bnd_idx ; 
    std::vector<float4> nrmt ; 
    std::vector<quad2>  prd ; 

    QPrd(); 
    void init();   
    void populate_prd(); 
    NP* duplicate_prd(int num_photon, int num_bounce ) const ; 

    std::string desc() const ; 
    int getNumBounce() const ; 

};




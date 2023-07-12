#pragma once
/**
SPrd : used by QSimTest/mock_propagate
=========================================

This was moved from QPrd 

Dummy per-ray-data enabling pure-CUDA (no OptiX, no geometry) 
testing of propagation using QSimTest MOCK_PROPAGATE.

The basis vectors of are obtained
from envvars SPRD_BND and SPRD_NRMT
which have defaults. 

    boundary_idx
    nrmt:(normals,distance) 

**/

#include <vector>
#include <string>

struct float4 ; 
struct quad2 ; 
struct SBnd ; 
struct NP ; 

struct SPrd
{
    static constexpr const bool VERBOSE = true ; 
    static constexpr const char* SPRD_NRMT_DEFAULT = "0,0,1,100 0,0,1,200 0,0,1,300 0,0,1,400" ; 
    static constexpr const char* SPRD_BND_DEFAULT = R"LITERAL(
    Acrylic///LS
    Water///Acrylic
    Water///Pyrex
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    )LITERAL" ;  

    // bnd name changed 
    // Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum

    const NP*           bnd ; 
    const SBnd*         sbnd ; 
    std::vector<int>    bnd_idx ; 
    std::vector<float4> nrmt ; 
    std::vector<quad2>  prd ; 

    SPrd(const NP* bnd); 
    void init();   
    void populate_prd(); 
    NP* duplicate_prd(int num_photon, int num_bounce ) const ; 

    std::string desc() const ; 
    int getNumBounce() const ; 
};



#include "vector_functions.h"
#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "NP.hh"
#include "SBnd.h"


inline SPrd::SPrd(const NP* bnd_)
    :
    bnd(bnd_),
    sbnd(bnd ? new SBnd(bnd) : nullptr)
{
    assert(sbnd); 
    init(); 
}


inline void SPrd::init()
{
    populate_prd(); 
}

/**
SPrd::populate_prd
--------------------

Sensitive to envvars SPRD_BND and SPRD_NRMT

**/

inline void SPrd::populate_prd()
{
    const char* bnd_sequence = ssys::getenvvar("SPRD_BND", SPRD_BND_DEFAULT );  
    if(VERBOSE) std::cerr << " SPRD_BND " << bnd_sequence << std::endl ; 
    sbnd->getBoundaryIndices( bnd_idx, bnd_sequence, '\n' ); 

    qvals( nrmt, "SPRD_NRMT", SPRD_NRMT_DEFAULT, true ); 

    int num_bnd_idx = bnd_idx.size() ; 
    int num_nrmt = nrmt.size() ; 

    bool consistent = num_bnd_idx == num_nrmt ; 
    std::cerr 
        << " SPrd::populate_prd "
        << " number of SPRD_BND mock boundaries "
        << " and SPRD_NRMT mock (normal,distance) must be the same "
        << " num_bnd_idx " << num_bnd_idx
        << " num_nrmt " << num_nrmt 
        << std::endl 
        ;
    assert(consistent); 

    int num_prd = num_bnd_idx ; 
    prd.resize(num_prd);  // vector of quad2
    for(int i=0 ; i < num_prd ; i++)
    {
        quad2& pr = prd[i] ; 
        pr.zero(); 
        pr.q0.f = nrmt[i] ; 
        pr.set_boundary( bnd_idx[i] ); 
        pr.set_identity( (i+1)*100 ); 
    }
}

/**
SPrd::duplicate_prd
---------------------

Duplicate the sequence of mock prd for all photon, 
if the num_bounce exceeds the prd obtained from environment 
the prd is wrapped within the photon.  

**/

inline NP* SPrd::duplicate_prd(int num_photon, int num_bounce) const 
{
    int num_prd = prd.size(); 
    int ni = num_photon ; 
    int nj = num_bounce ; 

    if(VERBOSE) std::cout 
        << "SPrd::duplicate_prd"
        << " ni:num_photon " << num_photon
        << " nj:num_bounce " << num_bounce
        << " num_prd " << num_prd 
        << std::endl 
        ;

    NP* a_prd = NP::Make<float>(ni, nj, 2, 4 ); 
    quad2* prd_v = (quad2*)a_prd->values<float>();  

    for(int i=0 ; i < ni ; i++)
        for(int j=0 ; j < nj ; j++) 
            prd_v[i*nj+j] = prd[j % num_prd] ; // wrap prd into array when not enough   

    return a_prd ; 
}

inline std::string SPrd::desc() const 
{
    std::stringstream ss ; 
    ss << "SPrd.sbn.descBoundaryIndices" << std::endl ; 
    ss << sbnd->descBoundaryIndices( bnd_idx ); 
    ss << "SPrd.nrmt" << std::endl ;  
    for(int i=0 ; i < int(nrmt.size()) ; i++ ) ss << nrmt[i] << std::endl ;  
    ss << "SPrd.prd" << std::endl ;  
    for(int i=0 ; i < int(prd.size()) ; i++ )  ss << prd[i].desc() << std::endl ;  
    std::string s = ss.str(); 
    return s ; 
}

inline int SPrd::getNumBounce() const 
{
    return bnd_idx.size(); 
}



#pragma once
/**
SPrd : used by QSimTest/mock_propagate
=========================================

This was moved from QPrd 

Dummy per-ray-data enabling pure-CUDA (no OptiX, no geometry) 
testing of propagation using QSimTest MOCK_PROPAGATE.

The basis vectors of are obtained from the below envvars 
which have constexpr defaults. 

SPRD_BND 
    boundary spec strings that that are converted into boundary_idx
    The CustomART PMT geometry simplification caused name change::

        Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
        Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum

SPRD_NRMT
    float4 strings parsed into nrmt:(normals,distance) 

SPRD_LPOSCOST
    space delimited floats with local intersect position cosine theta, range: -1. to 1.  
    This value is only relevant when mocking intersects onto special surfaces.

SPRD_IDENTITY
    space delimited ints with identity, which corresponds to sensor_identifier, aka lpmtid.
    This value is only relevant when mocking intersects onto special surfaces.

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
    static constexpr const char* SPRD_LPOSCOST_DEFAULT = "0,0,0,0.5" ; 
    static constexpr const char* SPRD_IDENTITY_DEFAULT = "0,0,0,1001" ; 
    static constexpr const char* SPRD_BND_DEFAULT = R"LITERAL(
    Acrylic///LS
    Water///Acrylic
    Water///Pyrex
    Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/NNVTMCPPMT_PMT_20inch_photocathode_mirror_logsurf/Vacuum
    )LITERAL" ;  
    
    const char*         bnd_sequence ; 
    const NP*           bnd ; 
    const SBnd*         sbnd ; 

    std::vector<float4> nrmt ; 
    std::vector<float>  lposcost; 
    std::vector<int >   identity ; 
    std::vector<int>    bnd_idx ; 

    std::vector<quad2>  prd ; 

    SPrd(const NP* bnd); 
    void init();   
    void init_evec();   
    void init_prd(); 

    std::string desc() const ; 
    int getNumBounce() const ; 

    NP* mock_prd(int num_photon, int num_bounce ) const ; 

};



#include "vector_functions.h"
#include "ssys.h"
#include "scuda.h"
#include "squad.h"
#include "NP.hh"
#include "SBnd.h"


inline SPrd::SPrd(const NP* bnd_)
    :
    bnd_sequence(ssys::getenvvar("SPRD_BND", SPRD_BND_DEFAULT )),
    bnd(bnd_),
    sbnd(bnd ? new SBnd(bnd) : nullptr)
{
    assert(sbnd); 
    init(); 
}

inline void SPrd::init()
{
    init_evec(); 
    init_prd(); 
}

/**
SPrd::init_evec
-----------------

Sensitive to envvars SPRD_BND, SPRD_NRMT, SPRD_IDENTITY, SPRD_LPOSCOST

**/

inline void SPrd::init_evec()
{
    sbnd->getBoundaryIndices( bnd_idx, bnd_sequence, '\n' ); 

    ssys::fill_evec<int>  (identity, "SPRD_IDENTITY", SPRD_IDENTITY_DEFAULT, ',' );  
    ssys::fill_evec<float>(lposcost, "SPRD_LPOSCOST", SPRD_LPOSCOST_DEFAULT, ',' );  

    qvals( nrmt, "SPRD_NRMT", SPRD_NRMT_DEFAULT, true ); 

    int num_bnd_idx  = bnd_idx.size() ; 
    int num_nrmt     = nrmt.size() ; 
    int num_identity = identity.size() ; 
    int num_lposcost = lposcost.size() ; 

    bool consistent = 
                      num_bnd_idx == num_nrmt  &&
                      num_bnd_idx == num_identity &&
                      num_bnd_idx == num_lposcost 
                    ; 

    if(!consistent) std::cerr 
        << "SPrd::init_evec : INCONSISTENT MOCKING "
        << " all four num MUST MATCH  "
        << std::endl 
        << desc() 
        ;
    assert(consistent); 
}



/**
SPrd::init_prd
--------------------

**/

inline void SPrd::init_prd()
{
    int num_prd = bnd_idx.size() ; 
    prd.resize(num_prd);  // vector of quad2

    for(int i=0 ; i < num_prd ; i++)
    {
        quad2& pr = prd[i] ; 
        pr.zero(); 

        pr.q0.f = nrmt[i] ; 
        pr.set_boundary( bnd_idx[i] ); 
        pr.set_identity( identity[i] ); 
        pr.set_lposcost( lposcost[i] ); 
    }
}



inline std::string SPrd::desc() const 
{
    std::stringstream ss ; 
    ss << "SPrd::desc"
        << " num_bnd_idx " << bnd_idx.size()
        << " num_nrmt " << nrmt.size()
        << " num_identity " << identity.size()
        << " num_lposcost " << lposcost.size()
        << std::endl 
        << ( bnd_sequence ? bnd_sequence : "-" )
        << std::endl 
        ;

    ss << "SPrd.sbn.descBoundaryIndices" << std::endl ; 
    ss << sbnd->descBoundaryIndices( bnd_idx ); 
    ss << "SPrd.nrmt" << std::endl ;  
    for(int i=0 ; i < int(nrmt.size()) ; i++ ) ss << nrmt[i] << std::endl ;  
    ss << "SPrd.prd" << std::endl ;  
    for(int i=0 ; i < int(prd.size()) ; i++ )  ss << prd[i].desc() << std::endl ;  
  
    std::string str = ss.str(); 
    return str ; 
}


inline int SPrd::getNumBounce() const 
{
    return bnd_idx.size(); 
}


/**
SPrd::mock_prd
----------------

Canonical use from QSimTest::mock_propagate

Duplicate the sequence of mock prd for all photon, 
if the num_bounce exceeds the prd obtained from environment 
the prd is wrapped within the photon bounces.   

**/

inline NP* SPrd::mock_prd(int num_photon, int num_bounce) const 
{
    int num_prd = prd.size(); 
    int ni = num_photon ; 
    int nj = num_bounce ; 

    if(VERBOSE) std::cout 
        << "SPrd::mock_prd"
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



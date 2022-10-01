#include "vector_functions.h"

#include "SSys.hh"
#include "scuda.h"
#include "squad.h"
#include "NP.hh"
#include "SLOG.hh"
#include "SEventConfig.hh"

#include "QBnd.hh"
#include "QPrd.hh"

#include "SBnd.h"


const plog::Severity QPrd::LEVEL = SLOG::EnvLevel("QPrd", "DEBUG") ; 

const QPrd* QPrd::INSTANCE = nullptr ; 
const QPrd* QPrd::Get(){ return INSTANCE ; }

QPrd::QPrd()
    :
    bnd(QBnd::Get()),
    sbn(bnd->sbn)
{
    if( bnd == nullptr )  LOG(fatal) << "QPrd must be instanciated after QBnd " ; 
    assert(bnd); 
    init(); 
    INSTANCE = this ; 
}

std::string QPrd::desc() const 
{
    std::stringstream ss ; 
    ss << "QPrd.sbn.descBoundaryIndices" << std::endl ; 
    ss << sbn->descBoundaryIndices( bnd_idx ); 
    ss << "QPrd.nrmt" << std::endl ;  
    for(unsigned i=0 ; i < nrmt.size() ; i++ ) ss << nrmt[i] << std::endl ;  
    ss << "QPrd.prd" << std::endl ;  
    for(unsigned i=0 ; i < prd.size() ; i++ )  ss << prd[i].desc() << std::endl ;  
    std::string s = ss.str(); 
    return s ; 
}


void QPrd::init()
{
    const char* bnd_fallback = "Acrylic///LS,Water///Acrylic,Water///Pyrex,Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum" ;  
    const char* bnd_sequence = SSys::getenvvar("QPRD_BND", bnd_fallback );  
    LOG(LEVEL) << " QPRD_BND " << bnd_sequence ; 
    sbn->getBoundaryIndices( bnd_idx, bnd_sequence, ',' ); 

    const char* nrmt_fallback = "0,0,1,100 0,0,1,200 0,0,1,300 0,0,1,400" ; 
    qvals( nrmt, "QPRD_NRMT", nrmt_fallback, true ); 

    unsigned num_prd = bnd_idx.size() ; 
    assert( num_prd == nrmt.size() ); 

    LOG(LEVEL) << " SEventConfig::Desc " << SEventConfig::Desc() ; 

    prd.resize(num_prd);  // vector of quad2
    for(unsigned i=0 ; i < num_prd ; i++)
    {
        quad2& pr = prd[i] ; 
        pr.zero(); 
        pr.q0.f = nrmt[i] ; 
        pr.set_boundary( bnd_idx[i] ); 
        pr.set_identity( (i+1)*100 ); 
    }
}

unsigned QPrd::getNumBounce() const 
{
    return bnd_idx.size(); 
}



/**
QPrd::duplicate_prd
---------------------

Duplicate the sequence of mock prd for all photon, 
if the num_bounce exceeds the prd obtained from environment 
the prd is wrapped within the photon.  

**/

NP* QPrd::duplicate_prd(unsigned num_photon, unsigned num_bounce) const 
{
    unsigned num_prd = prd.size(); 
    unsigned ni = num_photon ; 
    unsigned nj = num_bounce ; 
    LOG(LEVEL) 
        << " ni:num_photon " << num_photon
        << " nj:num_bounce " << num_bounce
        << " num_prd " << num_prd 
        ;


    NP* a_prd = NP::Make<float>(ni, nj, 2, 4 ); 

    quad2* prd_v = (quad2*)a_prd->values<float>();  
    for(unsigned i=0 ; i < ni ; i++)
    {
        for(unsigned j=0 ; j < nj ; j++) 
        {
             prd_v[i*nj+j] = prd[j % num_prd] ;   // wrap the prd if not enough   
        }
    }    
    return a_prd ; 
}





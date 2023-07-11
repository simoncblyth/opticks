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
    LOG_IF(fatal, bnd == nullptr ) << "QPrd must be instanciated after QBnd " ; 
    assert(bnd); 
    init(); 
    INSTANCE = this ; 
}


void QPrd::init()
{
    populate_prd(); 
}

/**
QPrd::populate_prd
--------------------

Sensitive to envvars QPRD_BND and QPRD_NRMT

**/

void QPrd::populate_prd()
{
    const char* bnd_sequence = SSys::getenvvar("QPRD_BND", QPRD_BND_DEFAULT );  
    LOG(LEVEL) << " QPRD_BND " << bnd_sequence ; 
    sbn->getBoundaryIndices( bnd_idx, bnd_sequence, '\n' ); 

    qvals( nrmt, "QPRD_NRMT", QPRD_NRMT_DEFAULT, true ); 

    int num_bnd_idx = bnd_idx.size() ; 
    int num_nrmt = nrmt.size() ; 

    bool consistent = num_bnd_idx == num_nrmt ; 
    LOG_IF(fatal, !consistent )    
        << " number of QPRD_BND mock boundaries "
        << " and QPRD_NRMT mock (normal,distance) must be the same "
        << " num_bnd_idx " << num_bnd_idx
        << " num_nrmt " << num_nrmt 
        ;
    assert(consistent); 


    LOG(LEVEL) << " SEventConfig::Desc " << SEventConfig::Desc() ; 

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
QPrd::duplicate_prd
---------------------

Duplicate the sequence of mock prd for all photon, 
if the num_bounce exceeds the prd obtained from environment 
the prd is wrapped within the photon.  

**/

NP* QPrd::duplicate_prd(int num_photon, int num_bounce) const 
{
    int num_prd = prd.size(); 
    int ni = num_photon ; 
    int nj = num_bounce ; 

    LOG(LEVEL) 
        << " ni:num_photon " << num_photon
        << " nj:num_bounce " << num_bounce
        << " num_prd " << num_prd 
        ;

    NP* a_prd = NP::Make<float>(ni, nj, 2, 4 ); 
    quad2* prd_v = (quad2*)a_prd->values<float>();  

    for(int i=0 ; i < ni ; i++)
        for(int j=0 ; j < nj ; j++) 
            prd_v[i*nj+j] = prd[j % num_prd] ; // wrap prd into array when not enough   

    return a_prd ; 
}

std::string QPrd::desc() const 
{
    std::stringstream ss ; 
    ss << "QPrd.sbn.descBoundaryIndices" << std::endl ; 
    ss << sbn->descBoundaryIndices( bnd_idx ); 
    ss << "QPrd.nrmt" << std::endl ;  
    for(int i=0 ; i < int(nrmt.size()) ; i++ ) ss << nrmt[i] << std::endl ;  
    ss << "QPrd.prd" << std::endl ;  
    for(int i=0 ; i < int(prd.size()) ; i++ )  ss << prd[i].desc() << std::endl ;  
    std::string s = ss.str(); 
    return s ; 
}

int QPrd::getNumBounce() const 
{
    return bnd_idx.size(); 
}



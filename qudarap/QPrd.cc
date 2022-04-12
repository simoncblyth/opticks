#include "vector_functions.h"

#include "SSys.hh"
#include "scuda.h"
#include "squad.h"
#include "NP.hh"
#include "PLOG.hh"
#include "SEventConfig.hh"

#include "QBnd.hh"
#include "QPrd.hh"

const QPrd* QPrd::INSTANCE = nullptr ; 
const QPrd* QPrd::Get(){ return INSTANCE ; }

QPrd::QPrd()
    :
    bnd(QBnd::Get())
{
    if( bnd == nullptr )  LOG(fatal) << "QPrd must be instanciated after QBnd " ; 
    assert(bnd); 
    init(); 
    INSTANCE = this ; 
}

void QPrd::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    bnd->dumpBoundaryIndices( bnd_idx ); 
    for(unsigned i=0 ; i < nrmt.size() ; i++ ) std::cout << nrmt[i] << std::endl ;  
    for(unsigned i=0 ; i < prd.size() ; i++ )  std::cout << prd[i].desc() << std::endl ;  
}

void QPrd::init()
{
    const char* bnd_fallback = "Acrylic///LS,Water///Acrylic,Water///Pyrex,Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum" ;  
    const char* bnd_sequence = SSys::getenvvar("QPRD_BND", bnd_fallback );  
    LOG(info) << " QPRD_BND " << bnd_sequence ; 
    bnd->getBoundaryIndices( bnd_idx, bnd_sequence, ',' ); 

    const char* nrmt_fallback = "0,0,1,100 0,0,1,200 0,0,1,300 0,0,1,400" ; 
    qvals( nrmt, "QPRD_NRMT", nrmt_fallback, true ); 

    unsigned num_prd = bnd_idx.size() ; 
    assert( num_prd == nrmt.size() ); 

    SEventConfig::SetMaxBounce( num_prd ); 
    SEventConfig::SetMaxRecord( num_prd+1 ); 
    LOG(info) << " SEventConfig::Desc " << SEventConfig::Desc() ; 

    prd.resize(num_prd); 
    for(unsigned i=0 ; i < num_prd ; i++)
    {
        quad2& pr = prd[i] ; 
        pr.zero(); 
        pr.q0.f = nrmt[i] ; 
        pr.set_boundary( bnd_idx[i] ); 
        pr.set_identity( (i+1)*100 ); 
    }
}



#include "vector_functions.h"

#include "SSys.hh"
#include "scuda.h"
#include "squad.h"
#include "NP.hh"
#include "PLOG.hh"

#include "QBnd.hh"
#include "QPrd.hh"

QPrd::QPrd(const QBnd* bnd_)
    :
    bnd(bnd_)
{
    init(); 
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
    const char* nrmt_fallback = "0,0,1,100 0,0,1,200 0,0,1,300 0,0,1,400" ; 

    const char* bnd_sequence = SSys::getenvvar("QPRD_BND", bnd_fallback );  
    LOG(info) << " QPRD_BND " << bnd_sequence ; 
    bnd->getBoundaryIndices( bnd_idx, bnd_sequence, ',' ); 

    qvals( nrmt, "QBND_NRMT", nrmt_fallback, true ); 

    assert( bnd_idx.size() == nrmt.size() ); 

    unsigned num_prd = bnd_idx.size() ; 
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




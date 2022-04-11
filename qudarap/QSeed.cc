
#include <cuda_runtime.h>
#include "scuda.h"
#include "squad.h"

#include "OpticksGenstep.h"

#include "QBuf.hh"
#include "QSeed.hh"
#include "QEvent.hh"

extern QBuf<int>* QSeed_create_photon_seeds(QBuf<float>* gs); 

QBuf<int>* QSeed::CreatePhotonSeeds(QBuf<float>* gs)  // static 
{
    return QSeed_create_photon_seeds(gs); 
}

void QSeed::CreatePhotonSeeds( QEvent* evt )
{
    evt->seed = QSeed_create_photon_seeds( evt->genstep ); 
}


/**
QSeed::ExpectedSeeds
----------------------

From a vector of counts populate the vector of seeds by simple CPU side duplication.  

**/

void QSeed::ExpectedSeeds(std::vector<int>& seeds,  unsigned& total, const std::vector<int>& counts ) // static 
{
    total = 0 ; 
    for(unsigned i=0 ; i < counts.size() ; i++)  total += counts[i] ; 

    unsigned ni = counts.size(); 
    for(unsigned i=0 ; i < ni ; i++)
    {
        int np = counts[i] ; 
        for(int p=0 ; p < np ; p++) seeds.push_back(i) ; 
    }
    assert( seeds.size() == total );  
}

int QSeed::CompareSeeds( const std::vector<int>& seeds, const std::vector<int>& xseeds ) // static 
{
    assert( seeds.size() == xseeds.size() ); 
    int mismatch = 0 ; 
    for(unsigned i=0 ; i < seeds.size() ; i++) if( seeds[i] != xseeds[i] ) mismatch += 1 ; 
    return mismatch ; 
}


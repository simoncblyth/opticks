#include <vector>
#include <cuda_runtime.h>
#include "scuda.h"
#include "QBuf.hh"
#include "QSeed.hh"
#include "QEvent.hh"

int main(int argc, char** argv)
{
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };
    std::vector<int> xseeds ; 
    unsigned xtotal ; 
    QSeed::ExpectedSeeds(xseeds, xtotal, photon_counts_per_genstep); 

    const NP* gs = QEvent::MakeFakeGensteps(photon_counts_per_genstep) ; 

    QEvent qe ; 
    qe.setGensteps(gs); 
    assert( qe.getNumPhotons() == xtotal ); 

    qe.seed->download_dump("qe.seed", 15 ); 

    std::vector<int> seeds ; 
    qe.seed->download(seeds); 

    int mismatch = QSeed::CompareSeeds( seeds, xseeds ); 
    std::cout << " mismatch " << mismatch << std::endl ; 
    assert( mismatch == 0 ); 

    return 0 ;  
}





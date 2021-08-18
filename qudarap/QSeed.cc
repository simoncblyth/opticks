
#include <cuda_runtime.h>
#include "scuda.h"
#include "SBuf.hh"
#include "QSeed.hh"

extern SBuf<int> QSeed_create_photon_seeds(SBuf<quad6> gs); 

SBuf<int> QSeed::CreatePhotonSeeds(SBuf<quad6> gs)
{
    return QSeed_create_photon_seeds(gs); 
}




#include "OPTICKS_LOG.hh"

#include <cuda_runtime.h>
#include "QBuf.hh"
#include "QEvent.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    QEvent* event = new QEvent ; 
    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    event->setGenstepsFake(photon_counts_per_genstep); 

    LOG(info) << event->desc() ; 
    event->seeds->download_dump("event->seeds", 10); 
    event->uploadEvt(); 
    event->checkEvt(); 

    cudaDeviceSynchronize(); 

    return 0 ; 
}


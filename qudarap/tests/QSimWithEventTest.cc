#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "SPath.hh"
#include "SSim.hh"
#include "NP.hh"
#include "SOpticksResource.hh"

#include <cuda_runtime.h>
#include "scuda.h"
#include "squad.h"
#include "sphoton.h"

#include "QSim.hh"
#include "SEvent.hh"
#include "QEvent.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const SSim* ssim = SSim::Load(); 
    QSim::UploadComponents(ssim); 

    QSim qs ; 

    QEvent event ; 

    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    unsigned x_total = 0 ; 
    for(unsigned i=0 ; i < photon_counts_per_genstep.size() ; i++) x_total += photon_counts_per_genstep[i] ; 

    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep) ; 

    event.setGenstep(gs); 
    assert( event.getNumPhoton() == x_total ); 

    LOG(info) << event.desc() ; 

    event.checkEvt(); 

    qs.generate_photon(&event); 

    // TODO: switch to NP
    std::vector<quad4> photon ; 
    event.downloadPhoton(photon); 
    LOG(info) << " downloadPhoton photon.size " << photon.size() ; 

    qs.dump_photon( photon.data(), photon.size(), "f0,f1,f2,i3" );  

    cudaDeviceSynchronize(); 

    return 0 ; 
}

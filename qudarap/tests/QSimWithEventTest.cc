#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "SPath.hh"
#include "SEvt.hh"
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

    SEvt evt ; 

    const SSim* ssim = SSim::Load(); 
    QSim::UploadComponents(ssim); 
    QSim qs ; 


    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    int x_total = 0 ; 
    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep, &x_total ) ; 
    SEvt::AddGenstep(gs); 


    qs.event->setGenstep(); 

    assert( int(qs.event->getNumPhoton()) == x_total ); 

    LOG(info) << qs.event->desc() ; 

    qs.event->checkEvt(); 

    qs.generate_photon();  



    NP* photon = qs.event->getPhoton();  
    photon->dump(); 


    cudaDeviceSynchronize(); 

    return 0 ; 
}

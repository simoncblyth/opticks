#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "SPath.hh"
#include "SEvt.hh"
#include "SSim.hh"
#include "NP.hh"

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

    SEvt* evt = SEvt::Create(SEvt::EGPU) ;
    assert( evt );  

    LOG(info) << "[ SSim::Load " ; 
    const SSim* sim = SSim::Load(); 
    LOG(info) << "] SSim::Load : sim " << sim  ; 

    LOG_IF(info, getenv("VERBOSE")!=nullptr ) 
         << "[sim.desc " 
         << std::endl 
         << sim->desc()
         << std::endl 
         << "]sim.desc " 
         ; 

    QSim::UploadComponents(sim); 

    QSim* qs = QSim::Create(); 
    QEvent* qev = qs->event ; 


    std::vector<int> photon_counts_per_genstep = { 3, 5, 2, 0, 1, 3, 4, 2, 4 };  
    int x_total = 0 ; 
    const NP* gs = SEvent::MakeCountGensteps(photon_counts_per_genstep, &x_total ) ; 
    SEvt::AddGenstep(gs); 

    qev->setGenstep(); 

    assert( int(qev->getNumPhoton()) == x_total ); 

    LOG(info) << qev->desc() ; 

    qev->checkEvt(); 

    qs->generate_photon();  


    NP* photon = qev->gatherPhoton();  
    photon->dump(); 

    cudaDeviceSynchronize(); 

    return 0 ; 
}

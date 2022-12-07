#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"

#include "OPTICKS_LOG.hh"
#include "stree.h"

#include "U4VolumeMaker.hh"
#include "U4Tree.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(error) << "[ U4VolumeMaker::PV " ; 
    const G4VPhysicalVolume* pv = U4VolumeMaker::PV();  // sensitive to GEOM envvar 
    LOG(error) << "] U4VolumeMaker::PV " ; 

    const G4String& pv_name = pv->GetName() ; 
    LOG(info) << " pv " << pv << " pv_name " << pv_name ; 

    stree st ; 
    U4Tree ut(&st, pv ) ;

    //LOG(info) << " st.desc " << st.desc() ; 

    int nn = st.get_num_nodes(); 
    for(int nidx=0 ; nidx < nn ; nidx++)
    {
        const char* so = st.get_soname(nidx); 
        std::cout
            << " nidx " << std::setw(6) << nidx 
            << " so " << so 
            << std::endl 
            ;
    }



    return 0 ; 
}

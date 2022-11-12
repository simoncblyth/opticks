#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SEvt.hh"
#include "CSGFoundry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SEvt* sev = SEvt::Load() ;  
    const char* cfbase = sev->getSearchCFBase() ; // search up dir tree starting from loaddir for dir with CSGFoundry/solid.npy

    SSim::Create();  
    const CSGFoundry* fd = CSGFoundry::Load(cfbase);
    sev->setGeo(fd); 

    int ins_idx = SSys::getenvint("INS_IDX", 39216) ;
    if( ins_idx >= 0 ) sev->setFrame(ins_idx); 
    std::cout << sev->descFull() ; 
  
    return 0 ; 
}


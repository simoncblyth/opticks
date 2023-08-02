#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SEvt.hh"
#include "CSGFoundry.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    LOG(info) << "[ SEvt::Load" ; 
    SEvt* sev = SEvt::LoadRelative() ;  
    LOG(info) << sev->descFold(); 
    int total_items = sev->getTotalItems() ; 

    LOG(info) << " total_items " << total_items ; 
    if(total_items == 0 ) return 0 ; 


    LOG(info) << "] SEvt::Load" ; 

    LOG(info) << " loaded SEvt from " << sev->getLoadDir() ;  
    LOG(info) << sev->desc() ; 

    if(sev->is_loadfail)
    {
        LOG(info) << " sev.is_loadfail" ; 
        return 0 ; 
    }

    const char* cfbase = sev->getSearchCFBase() ; 
    // search up dir tree starting from loaddir for dir with CSGFoundry/solid.npy

    SSim::Create();  
    const CSGFoundry* fd = CSGFoundry::Load(cfbase);
    sev->setGeo(fd); 

    int ins_idx = SSys::getenvint("INS_IDX", 39216) ;
    if( ins_idx >= 0 ) sev->setFrame(ins_idx); 
    std::cout << sev->descFull() ; 
  
    return 0 ; 
}


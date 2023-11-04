#include "OPTICKS_LOG.hh"
#include "spath.h"
#include "NPFold.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* base = "$TMP/U4Material_MakePropertyFold" ; 
    const char* name = "U4Material" ; 
    const char* fold = spath::Resolve(base, name) ; 

    NPFold* mats = NPFold::LoadIfExists(fold) ; 

    LOG_IF(error, mats==nullptr) << " NO NPFold AT " << fold ; 
    if( mats == nullptr ) return 0 ; 

    LOG(info) << std::endl << mats->desc_subfold(name) << std::endl ; 
    LOG(info) << " fold [" << fold << "]" ; 

    const char* s = "Water" ; 
    const NPFold* f = mats->find_subfold(s) ; 
    if(!f) return 0 ; 
    LOG(info) << " f->desc_subfold(" << s << ")" << std::endl << f->desc() << std::endl ;   

    return 0 ; 
}

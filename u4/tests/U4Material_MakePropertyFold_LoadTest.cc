#include "SPath.hh"
#include "OPTICKS_LOG.hh"
#include "NPFold.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* base = "$TMP/U4Material_MakePropertyFold" ; 
    const char* name = "U4Material" ; 
    const char* fold = SPath::Resolve(base, name, NOOP ) ; 

    NPFold* mats = NPFold::Load(fold) ; 
    LOG(info) << std::endl << mats->desc_subfold(name) << std::endl ; 

    LOG(info) << " fold [" << fold << "]" ; 

    const char* s = "Water" ; 
    const NPFold* f = mats->find_subfold(s) ; 
    LOG(info) << " f->desc_subfold(" << s << ")" << std::endl << f->desc() << std::endl ;   

    return 0 ; 
}

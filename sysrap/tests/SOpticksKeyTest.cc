#include "OPTICKS_LOG.hh"
#include "SOpticksKey.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    SOpticksKey::SetKey(nullptr); 
    SOpticksKey* key = SOpticksKey::GetKey(); 

    if( key == nullptr ) 
    {
        LOG(error) << " key NULL " ; 
    }
    else 
    {
        LOG(info) << key->desc() ;      
    }

    return 0 ; 
}


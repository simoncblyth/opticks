// TEST=BResourceTest om-t 

#include "OPTICKS_LOG.hh"
#include "BOpticksResource.hh"
#include "BResource.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    BOpticksResource bor ; 

    const char* key = argc > 1 ? argv[1] : "tmpuser_dir" ; 
    const char* val = BResource::Get(key) ; 

    LOG(info) 
        << " key " << key 
        << " val " << val
        ; 


    BResource::Dump("BResourceTest"); 

    return 0 ; 
}

// TEST=BResourceTest om-t 

#include "OPTICKS_LOG.hh"
#include "BResource.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* key = argc > 1 ? argv[1] : "tmpuser_dir" ; 
    const char* val = BResource::Get(key) ; 

    LOG(info) 
        << " key " << key 
        << " val " << val
        ; 

    return 0 ; 
}

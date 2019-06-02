// TEST=BResourceTest om-t 

#include "OPTICKS_LOG.hh"
#include "BOpticksResource.hh"
#include "BResource.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    BOpticksResource br ; 

    const char* key = argc > 1 ? argv[1] : "tmpuser_dir" ; 
    const char* nval = argc > 2 ? argv[2] : "/tmp" ; 
    const char* val = BResource::GetDir(key) ; 

    LOG(info) 
        << " key " << key 
        << " val " << val
        << " nval " << nval
        ; 


    BResource::Dump("BResourceTest.0"); 
    BResource::SetDir(key, nval) ; 
    BResource::Dump("BResourceTest.1"); 

    return 0 ; 
}

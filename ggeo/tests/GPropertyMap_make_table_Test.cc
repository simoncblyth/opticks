
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "GPropertyMap.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* path = SSys::getenvvar("GPropertyMap_BASE"); 
    if(path == nullptr) return 0 ; 

    const char* name = SSys::getenvvar("GPropertyMap_NAME", "LS" ); 
    const char* type = SSys::getenvvar("GPropertyMap_TYPE", "material" ); 

    GPropertyMap<double>* pmap = GPropertyMap<double>::load( path, name, type ); 

    LOG(info) << pmap->desc_table() ; 
    LOG(info) << pmap->make_table() ; 


    return 0 ; 
}

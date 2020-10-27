#include "OPTICKS_LOG.hh"
#include "SensorLib.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    
    const char* dir = "$TMP/opticksgeo/SensorLib" ; 
    SensorLib* sl = SensorLib::Load(dir); 
  
    assert( sl); 

    return 0 ; 
}

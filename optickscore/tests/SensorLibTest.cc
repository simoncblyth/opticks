#include "OPTICKS_LOG.hh"
#include "SensorLib.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    
    const char* dir = "$TMP/optickscore/SensorLib" ; 
    SensorLib* sl = SensorLib::Load(dir); 
    if(sl == NULL) 
    {
        LOG(fatal) << " failed to load from " << dir ; 
        return 0 ; 
    } 
    sl->dump("SensorLibTest");

    return 0 ; 
}

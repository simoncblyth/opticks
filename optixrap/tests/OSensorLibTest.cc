#include "SensorLib.hh"
#include "OSensorLib.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* dir = "$TMP/opticksgeo/MockSensorLibTest" ;

    SensorLib* senlib = SensorLib::Load(dir); 

    if( senlib == NULL )
    {
        LOG(fatal) << " FAILED to load from " << dir ; 
        return 0 ;
    }

    senlib->dump("OSensorLibTest"); 

    optix::Context context = optix::Context::create(); 

    OSensorLib osenlib(context, senlib);    

    osenlib.convert(); 


    return 0 ; 
}



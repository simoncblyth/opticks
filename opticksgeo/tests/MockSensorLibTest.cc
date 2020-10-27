#include "OPTICKS_LOG.hh"
#include "SensorLib.hh"
#include "MockSensorLib.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned num_sensor = 100 ; 

    SensorLib* senlib = MockSensorLib::Make(num_sensor); 

    assert( senlib->getNumSensor() == num_sensor ); 
  
    const char* dir = "$TMP/opticksgeo/tests/MockSensorLibTest" ; 

    senlib->save( dir );  

    SensorLib* senlib2 = SensorLib::Load(dir) ;  

    if( senlib2 == NULL )
    {
        LOG(fatal) << " FAILED to load from " << dir ; 
        return 0 ; 
    }

    assert( senlib2->getNumSensor() == num_sensor ); 

    senlib2->dump("MockSensorLibTest"); 

    LOG(info) << dir ; 

    return 0 ; 
}

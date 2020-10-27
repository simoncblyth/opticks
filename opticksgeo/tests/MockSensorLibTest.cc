#include "OPTICKS_LOG.hh"
#include "SensorLib.hh"
#include "MockSensorLib.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned num_sensor = 100 ; 

    SensorLib* senlib = MockSensorLib::Make(num_sensor); 

    assert( senlib->getNumSensor() == num_sensor ); 
  
    const char* dir = "$TMP/opticksgeo/MockSensorLibTest" ; 

    senlib->save( dir );  

    SensorLib* senlib2 = SensorLib::Load(dir) ;  

    assert( senlib2->getNumSensor() == num_sensor ); 

    senlib2->dump("MockSensorLibTest"); 

    return 0 ; 
}

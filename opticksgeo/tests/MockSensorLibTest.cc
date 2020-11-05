#include "OPTICKS_LOG.hh"
#include "SensorLib.hh"
#include "MockSensorLib.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned num_cat = 2 ; 
    //unsigned num_cat = 0 ;        // model the case of having no angular efficiency 
    unsigned num_sensor = 10 ;    // can still have simple sensor info without the angular efficiency   

    SensorLib* senlib = MockSensorLib::Make(num_cat, num_sensor); 

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
    LOG(info) << senlib->desc()  ; 
    LOG(info) << senlib2->desc()  ; 


    senlib2->close();  


    return 0 ; 
}

#include "OPTICKS_LOG.hh"
#include "SensorLib.hh"
#include "SphereOfTransforms.hh"
#include "MockSensorLib.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    // match OSensorLibGeoTest to mock the corresponding number of sensors 
    unsigned num_theta = 64+1 ; 
    unsigned num_phi   = 128 ; 
    unsigned num_sensor = SphereOfTransforms::NumTransforms(num_theta, num_phi); 

    unsigned num_cat = 2 ; 
    //unsigned num_cat = 0 ;        // model the case of having no angular efficiency 

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

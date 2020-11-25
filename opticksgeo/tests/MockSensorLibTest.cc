#include "OPTICKS_LOG.hh"
#include "SensorLib.hh"
#include "SphereOfTransforms.hh"
#include "MockSensorLib.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned num_cat = argc > 1 ? atoi(argv[1]) : 2 ;   // use 0 to model the case of having no angular efficiency 

    // match OSensorLibGeoTest to mock the corresponding number of sensors 
    unsigned num_theta = 64+1 ; 
    unsigned num_phi   = 128 ; 
    unsigned num_sensor = SphereOfTransforms::NumTransforms(num_theta, num_phi); 

    LOG(info) 
        << " num_theta " << num_theta
        << " num_phi " << num_phi
        << " num_sensor " << num_sensor
        << " num_cat " << num_cat
        << ( num_cat == 0 ? " NO ANGULAR EFFICIENCY " : " with angular efficiency " )
        ;


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

    unsigned modulo = 100 ; 
    senlib2->dump("MockSensorLibTest", modulo); 

    LOG(info) << dir ; 
    LOG(info) << senlib->desc()  ; 
    LOG(info) << senlib2->desc()  ; 


    senlib2->close();  


    return 0 ; 
}
// om-;TEST=MockSensorLibTest om-t

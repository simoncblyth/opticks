#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "MockSensorAngularEfficiencyTable.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    unsigned num_sensor_cat  = 1 ; 
    unsigned num_theta_steps = 180 ;  // height
    unsigned num_phi_steps   = 360 ;  // width 

    NPY<float>* tab = MockSensorAngularEfficiencyTable::Make(num_sensor_cat, num_theta_steps, num_phi_steps); 

    const char* path = "$TMP/optickscore/tests/MockSensorAngularEfficiencyTableTest.npy" ;
    LOG(info) << " save to " << path ;  
    tab->save(path); 

    tab->dump();

    return 0 ; 
}

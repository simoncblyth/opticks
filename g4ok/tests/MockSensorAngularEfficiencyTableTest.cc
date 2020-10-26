#include "OPTICKS_LOG.hh"
#include "NPY.hpp"
#include "MockSensorAngularEfficiencyTable.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    unsigned num_cat = 1 ; 
    unsigned num_theta_steps = 181 ;  // height
    unsigned num_phi_steps = 361 ;    // width 

    MockSensorAngularEfficiencyTable tab( num_cat, num_theta_steps, num_phi_steps ); 
  
    NPY<float>* arr = tab.getArray(); 

    const char* path = "$TMP/G4OK/tests/MockSensorAngularEfficiencyTableTest.npy" ;
    LOG(info) << " save to " << path ;  
    arr->save(path); 

    return 0 ; 
}

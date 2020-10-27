#include "SensorLib.hh"
#include "MockSensorLib.hh"
#include "MockSensorAngularEfficiencyTable.hh"

SensorLib* MockSensorLib::Make(unsigned num_sensor)  // static
{
    MockSensorLib msl(num_sensor); 
    return msl.getSensorLib(); 
}

MockSensorLib::MockSensorLib(unsigned num_sensor)
    :
    m_sensorlib(new SensorLib)
{
    initSensorData(num_sensor); 
    initAngularEfficiency( 180, 360 );  // height:theta, width:phi
}

void MockSensorLib::initSensorData(unsigned num_sensor)
{
    m_sensorlib->initSensorData(num_sensor); 
    for(unsigned i=0 ; i < num_sensor ; i++)
    {
        unsigned sensor_index = i ; 
        unsigned sensor_identifier = 1000000 + i ; 
        float efficiency_1 = 0.5f ;    
        float efficiency_2 = 1.0f ;    
        int   sensor_cat = 0 ; 
        m_sensorlib->setSensorData( sensor_index, efficiency_1, efficiency_2, sensor_cat, sensor_identifier );
    }   
}

void MockSensorLib::initAngularEfficiency(unsigned num_theta_steps, unsigned num_phi_steps)
{
    unsigned num_cat = 1 ; 
    MockSensorAngularEfficiencyTable tab( num_cat, num_theta_steps, num_phi_steps );  // cat, height, width
    NPY<float>* angular_efficiency = tab.getArray(); 
    m_sensorlib->setSensorAngularEfficiency( angular_efficiency ); 
}

SensorLib* MockSensorLib::getSensorLib() const 
{
   return m_sensorlib ; 
}



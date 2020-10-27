#pragma once

#include "plog/Severity.h"
class SensorLib ; 

#include "OKGEO_API_EXPORT.hh"

class OKGEO_API MockSensorLib
{
    private:
        static const plog::Severity LEVEL ;  
    public:
        static SensorLib* Make(unsigned num_sensor);
    public:
        MockSensorLib(unsigned num_sensor); 
        SensorLib* getSensorLib() const ; 
    private:
        void initSensorData(unsigned num_sensor); 
        void initAngularEfficiency(unsigned num_theta_steps, unsigned num_phi_steps);
    private:
        SensorLib* m_sensorlib ;  

};


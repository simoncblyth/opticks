#pragma once

#include "plog/Severity.h"
class SensorLib ; 

#include "OKCORE_API_EXPORT.hh"

class OKCORE_API MockSensorLib
{
    private:
        static const plog::Severity LEVEL ;  
    public:
        static SensorLib* Make(unsigned num_cat, unsigned num_sensor);
    public:
        MockSensorLib(unsigned num_cat, unsigned num_sensor); 
        SensorLib* getSensorLib() const ; 
    private:
        void initSensorData(unsigned num_sensor); 
        void initAngularEfficiency(unsigned num_theta_steps, unsigned num_phi_steps);
    private:
        unsigned   m_num_cat ; 
        SensorLib* m_sensorlib ;  

};


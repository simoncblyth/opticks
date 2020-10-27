#pragma once 

#include <vector>
#include "plog/Severity.h"
template <typename T> class NPY ; 

#include "OKGEO_API_EXPORT.hh"

class OKGEO_API SensorLib 
{
    private:
        static const plog::Severity LEVEL ;  
        static const char* SENSOR_DATA ;
        static const char* SENSOR_ANGULAR_EFFICIENCY ;

        static const NPY<float>*  
            MakeSensorAngularEfficiency( const std::vector<int>& shape, const std::vector<float>& values,
                                         int theta_steps=180, float theta_min=0.f, float theta_max=180.f, 
                                         int phi_steps=1,     float phi_min=0.f, float phi_max=360.f );
    public: 
        static SensorLib* Load(const char* dir);  
    public: 
        SensorLib(const char* dir=NULL);
        void initSensorData(unsigned sensor_num ); 
        void setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int sensor_category, int sensor_identifier);
    public: 
        void getSensorData(unsigned sensorIndex, float& efficiency_1, float& efficiency_2, int& category, int& identifier) const ;
        int  getSensorIdentifier(unsigned sensorIndex) const ;    

    public: 
        void setSensorAngularEfficiency( const std::vector<int>& shape, const std::vector<float>& values,
                                         int theta_steps=180, float theta_min=0.f, float theta_max=180.f,
                                         int phi_steps=1,     float phi_min=0.f, float phi_max=360.f );

        void setSensorAngularEfficiency( const NPY<float>* sensor_angular_efficiency );

    public: 
        void save(const char* dir) const ;

    private:
        bool          m_loaded ; 
        NPY<float>*   m_sensor_data ; 
        unsigned      m_sensor_num ; 

        const NPY<float>*   m_sensor_angular_efficiency ; 
};



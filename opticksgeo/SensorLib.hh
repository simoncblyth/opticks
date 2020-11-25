#pragma once 

#include <vector>
#include <map>
#include <string>

#include "plog/Severity.h"
template <typename T> class NPY ; 

#include "OKGEO_API_EXPORT.hh"

/**
SensorLib
===========

Canonically instanciated within G4Opticks::setGeometry 
at which point initSensorData sets m_num_sensor

**/

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
        std::string desc() const ;

        void initSensorData(unsigned sensor_num ); 
        void setSensorData(unsigned sensorIndex, float efficiency_1, float efficiency_2, int sensor_category, int sensor_identifier);
    public: 
        unsigned getNumSensor() const ;
        void getSensorData(unsigned sensorIndex, float& efficiency_1, float& efficiency_2, int& category, int& identifier) const ;
        int  getSensorIdentifier(unsigned sensorIndex) const ;    

    public: 
        void setSensorAngularEfficiency( const std::vector<int>& shape, const std::vector<float>& values,
                                         int theta_steps=180, float theta_min=0.f, float theta_max=180.f,
                                         int phi_steps=1,     float phi_min=0.f, float phi_max=360.f );

        void setSensorAngularEfficiency( const NPY<float>* sensor_angular_efficiency );
        unsigned getNumSensorCategories() const ;
    public: 
        bool isClosed() const ;
        void close();   // needs to be invoked after sensorlib data collection is completed
    private: 
        void checkSensorCategories(bool dump);
        void dumpCategoryCounts(const char* msg="SensorLib::dumpCategoryCounts") const ;
    public: 
         // needed for OSensorLib
         NPY<float>*        getSensorDataArray() const;
         const NPY<float>*  getSensorAngularEfficiencyArray() const;
    public: 
        void save(const char* dir) const ;
        void dump(const char* msg="SensorLib::dump", unsigned modulo=0) const ;
        void dumpSensorData(const char* msg, unsigned modulo=0) const ;
        void dumpAngularEfficiency(const char* msg) const ;
    private:
        bool                m_loaded ; 
        NPY<float>*         m_sensor_data ; 
        unsigned            m_sensor_num ; 
        const NPY<float>*   m_sensor_angular_efficiency ; 
        bool                m_closed ;  
        std::map<int,int>   m_category_counts ; 
};



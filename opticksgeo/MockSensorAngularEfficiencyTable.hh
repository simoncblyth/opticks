#pragma once

#include <vector>
#include "plog/Severity.h"
template <typename T> class NPY ; 

#include "OKGEO_API_EXPORT.hh"

struct OKGEO_API MockSensorAngularEfficiencyTable
{
    static const plog::Severity LEVEL ;  
    static NPY<float>* Make(unsigned sensor_cat, unsigned theta_steps, unsigned phi_steps); 

    unsigned m_num_sensor_cat ;  
    unsigned m_num_theta_steps ; 
    unsigned m_num_phi_steps ; 
    unsigned m_multiplicity ; 

    float m_theta_min ;     
    float m_theta_max ;     
    float m_theta_step ; 

    float m_phi_min ;     
    float m_phi_max ;     
    float m_phi_step ; 

    std::vector<int>   m_shape  ;    // (sensor_categories, phi_steps, theta_steps) 
    std::vector<float> m_values ; 
    NPY<float>*        m_array ; 

    // when no phi-dependency use phi_steps=1 
    MockSensorAngularEfficiencyTable(unsigned sensor_cat, unsigned theta_steps, unsigned phi_steps=1 );
    void init() ; 


    float getEfficiency(unsigned i_cat, unsigned j_theta, unsigned k_phi) const ;

    NPY<float>* getArray() const ; 
    const std::vector<int>   getShape() const ; 
    const std::vector<float> getValues() const ; 


};




#pragma once

#ifdef __CUDACC__
/**
OSensorLib_angular_efficiency
-------------------------------

Interpolated lookup of sensor angular efficiency for local angle fractions and sensor category.

category
    0-based sensor category index, typically for a small number (<10) of sensor types
    Category -1 corresponds to the case when no angular efficiency is available for 
    a sensor resulting in a returned angular_efficiency of 1.f 
  
phi_fraction
    fraction of the 360 degree azimuthal phi angle, range 0. to 1. 
    (when there is no phi-dependence of efficiency use 0. for clarity)

theta_fraction
    fraction of the 180 degree polar theta angle, range 0. to 1. 

The sensor efficiency textures for each category are constructed by OSensorLib::convert
from the array held by okg/SensorLib 

See tests/OSensorLibTest 
**/

rtBuffer<int4,1>  OSensorLib_texid ;

static __device__ __inline__ float OSensorLib_angular_efficiency(int category, float phi_fraction, float theta_fraction  )
{
    int tex_id = category > -1 ? OSensorLib_texid[category].x : -1 ; 
    float angular_efficiency = tex_id > -1 ? rtTex2D<float>( tex_id, phi_fraction, theta_fraction ) : 1.f ;  
    return angular_efficiency ; 
}

#else

#include "OXPPNS.hh"

class SensorLib ; 
class OCtx ; 

template <typename T> class NPY ;

#include "plog/Severity.h"
#include "OXRAP_API_EXPORT.hh"

/**
OSensorLib
===========

**/

class OXRAP_API OSensorLib 
{
    public:
        static const plog::Severity LEVEL ; 
        static const char*  TEXID ; 
    public:
        OSensorLib(const OCtx* octx, const SensorLib* sensorlib);
        const OCtx* getOCtx() const ;
    private:
        void init();
    public:
        const NPY<float>*  getSensorAngularEfficiencyArray() const ;
        unsigned getNumSensorCategories() const ;
        unsigned getNumTheta() const ;
        unsigned getNumPhi() const ;
        unsigned getNumElem() const ;
    public:
        int      getTexId(unsigned icat) const ;
    public:
        void convert();
    private:    
        void makeSensorAngularEfficiencyTexture();
    private:    
        const OCtx*        m_octx ; 
        const SensorLib*   m_sensorlib ; 
        const NPY<float>*  m_sensor_data ; 
        const NPY<float>*  m_angular_efficiency ; 

        unsigned           m_num_dim ; 
        unsigned           m_num_cat ;
        unsigned           m_num_theta ;
        unsigned           m_num_phi ;
        unsigned           m_num_elem ;
        NPY<int>*          m_texid ; 

};

#endif

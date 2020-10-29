#pragma once

#ifdef __CUDACC__
/**
OSensorLib_angular_efficiency
-------------------------------

Interpolated lookup of sensor angular efficiency for local angle fractions and sensor category.

category
    0-based sensor category index, typically for a small number (<10) of sensor types  
phi_fraction
    fraction of the 360 degree azimuthal phi angle, range 0. to 1. 
theta_fraction
    fraction of the 180 degree polar theta angle, range 0. to 1. 

The sensor efficiency textures for each category are constructed by OSensorLib::convert
from the array held by okg/SensorLib 

See tests/OSensorLibTest 
**/

rtBuffer<int4,1>  OSensorLib_texid ;

static __device__ __inline__ float OSensorLib_angular_efficiency(int category, float phi_fraction, float theta_fraction  )
{
    int tex_id = OSensorLib_texid[category].x ; 
    float angular_efficiency = rtTex2D<float>( tex_id, phi_fraction, theta_fraction );  
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
        const NPY<float>*  m_angular_efficiency ; 

        unsigned           m_num_dim ; 
        unsigned           m_num_cat ;
        unsigned           m_num_theta ;
        unsigned           m_num_phi ;
        unsigned           m_num_elem ;
        NPY<int>*          m_texid ; 

};

#endif

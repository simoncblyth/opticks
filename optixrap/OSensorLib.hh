#pragma once

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
public:
    OSensorLib(OCtx* octx, SensorLib* sensorlib);
    OCtx* getOCtx() const ;
public:
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
    OCtx*              m_octx ; 
    SensorLib*         m_sensorlib ; 
    const NPY<float>*  m_angular_efficiency ; 
    unsigned           m_num_dim ; 
    unsigned           m_num_cat ;
    unsigned           m_num_theta ;
    unsigned           m_num_phi ;
    unsigned           m_num_elem ;
    int*               m_tex_id ; 


};



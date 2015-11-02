#include "GSolid.hh"
#include "GPropertyMap.hh"
#include "GMesh.hh"

#include "GGeo.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"

//#include "GBoundary.hh"

// npy-
#include "NSensor.hpp"

#include "stdio.h"

void GSolid::init()
{
    m_blib = m_ggeo->getBndLib(); 
    m_slib = m_ggeo->getSurfaceLib(); 
}

void GSolid::Summary(const char* msg )
{
   if(!msg) msg = getDescription();
   if(!msg) msg = "GSolid::Summary" ;
   printf("%s\n", msg );
}


void GSolid::setBoundary(unsigned int boundary)
{
    m_boundary = boundary ; 
    setBoundaryIndices( boundary );
}


void GSolid::setSensor(NSensor* sensor)
{
    m_sensor = sensor ; 
    // every triangle needs a value... use 0 to mean unset, so sensor   
    setSensorIndices( NSensor::RefIndex(sensor) );
}


unsigned int GSolid::getSensorSurfaceIndex()
{
    // sensor indices are set even for non sensitive volumes in PMT viscinity
    // TODO: change that 
    // this is a workaround that requires an associated sensitive surface
    // in order for the index to be provided

    //bool oss = m_boundary ? m_boundary->hasOuterSensorSurface() : false ; 
    //unsigned int ssi = oss ? NSensor::RefIndex(m_sensor) : 0 ;  
    //return ssi ; 

    unsigned int surface = m_blib->getOuterSurface(m_boundary);
    bool oss = m_slib->isSensorSurface(surface); 
    unsigned int ssi = oss ? NSensor::RefIndex(m_sensor) : 0 ;  
    return ssi ; 
}

guint4 GSolid::getIdentity()
{
    return guint4(
                   m_index, 
                   m_mesh ? m_mesh->getIndex() : 0, 
                   m_boundary,
                   getSensorSurfaceIndex()
                 );
}
 




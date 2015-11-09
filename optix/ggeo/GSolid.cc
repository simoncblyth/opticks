#include "GSolid.hh"
#include "GPropertyMap.hh"
#include "GMesh.hh"

#include "GGeo.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"

// npy-
#include "NSensor.hpp"

#include "stdio.h"

void GSolid::Summary(const char* msg )
{
   if(!msg) msg = getDescription();
   if(!msg) msg = "GSolid::Summary" ;
   printf("%s\n", msg );
}


std::string GSolid::description()
{
   return getDescription(); 
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

guint4 GSolid::getIdentity()
{
    return guint4(
                   m_index, 
                   m_mesh ? m_mesh->getIndex() : 0, 
                   m_boundary,
                   getSensorSurfaceIndex()
                 );
}
 




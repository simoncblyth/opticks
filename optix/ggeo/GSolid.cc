#include "GSolid.hh"
#include "GPropertyMap.hh"
#include "GMesh.hh"
#include "GBoundary.hh"
#include "GSensor.hh"

#include "stdio.h"

void GSolid::Summary(const char* msg )
{
   if(!msg) msg = getDescription();
   if(!msg) msg = "GSolid::Summary" ;
   printf("%s\n", msg );
}


void GSolid::setBoundary(GBoundary* boundary)
{
    m_boundary = boundary ; 
    setBoundaryIndices( boundary->getIndex() );
}

void GSolid::setSensor(GSensor* sensor)
{
    m_sensor = sensor ; 
    // every triangle needs a value... use 0 to mean unset, so sensor   
    setSensorIndices( GSensor::RefIndex(sensor) );
}


guint4 GSolid::getIdentity()
{
    return guint4(
                   m_index, 
                   m_mesh ? m_mesh->getIndex() : 0, 
                   m_boundary ? m_boundary->getIndex() : 0,
                   GSensor::RefIndex(m_sensor)
                 );
}
 

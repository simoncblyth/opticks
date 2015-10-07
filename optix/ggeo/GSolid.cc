#include "GSolid.hh"
#include "GPropertyMap.hh"
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




 

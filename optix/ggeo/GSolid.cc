#include "GSolid.hh"
#include "GPropertyMap.hh"
#include "GSubstance.hh"

#include "stdio.h"

void GSolid::Summary(const char* msg )
{
   if(!msg) msg = getDescription();
   if(!msg) msg = "GSolid::Summary" ;
   printf("%s\n", msg );
}


void GSolid::setSubstance(GSubstance* substance)
{
    m_substance = substance ; 
    setSubstanceIndices( substance->getIndex() );
}

 

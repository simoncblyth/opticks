#include "GSolid.hh"
#include "GPropertyMap.hh"
#include "GSubstance.hh"

#include "stdio.h"

GSolid::GSolid( unsigned int index, GMatrixF* transform, GMesh* mesh, GSubstance* substance)
         : 
         GNode(index, transform, mesh ),
         m_substance(substance),
         m_selected(true)
{
    // NB not taking ownership yet 
}

GSolid::~GSolid()
{
}

void GSolid::setSelected(bool selected)
{
    m_selected = selected ; 
}
bool GSolid::isSelected()
{
   return m_selected ; 
}


void GSolid::Summary(const char* msg )
{
   if(!msg) msg = getDescription();
   if(!msg) msg = "GSolid::Summary" ;
   printf("%s\n", msg );
}


void GSolid::setSubstance(GSubstance* substance)
{
    m_substance = substance ; 
    setMeshSubstance( substance->getIndex() );
}

GSubstance* GSolid::getSubstance()
{
    return m_substance ; 
}


 

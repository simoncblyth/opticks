#include <cstring>
#include "CSurfaceTable.hh"

CSurfaceTable::CSurfaceTable(const char* name)
   :
   m_name(strdup(name))
{
}

const char* CSurfaceTable::getName()
{
    return m_name ; 
}

void CSurfaceTable::add(const G4OpticalSurface* surf)
{
    m_surfaces.push_back(surf);
}

unsigned CSurfaceTable::getNumSurf()
{
    return m_surfaces.size();
}

const G4OpticalSurface* CSurfaceTable::getSurface(unsigned index)
{
    return index < m_surfaces.size() ? m_surfaces[index] : NULL ; 
}

 

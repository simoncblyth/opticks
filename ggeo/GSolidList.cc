#include "GSolidList.hh"


GSolidList::GSolidList()
{
}

unsigned GSolidList::getNumSolids()
{
    return m_solids.size();
}

GSolid* GSolidList::getSolid(unsigned index)
{
    return m_solids[index] ; 
}

void GSolidList::add( GSolid* solid )
{
    m_solids.push_back(solid);
}

std::vector<GSolid*>& GSolidList::getList()
{
    return m_solids ; 
}


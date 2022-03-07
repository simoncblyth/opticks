#include "SOpticks.hh"

bool SOpticks::isEnabledMergedMesh(unsigned ridx)
{
    return true ; 
}

std::vector<unsigned>&  SOpticks::getSolidSelection() 
{
    return m_solid_selection ; 
}

const std::vector<unsigned>&  SOpticks::getSolidSelection() const 
{
    return m_solid_selection ; 
}



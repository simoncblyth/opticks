#pragma once

#include <vector>
#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SOpticks
{
    bool  isEnabledMergedMesh(unsigned ridx); 

    std::vector<unsigned>&        getSolidSelection() ;
    const std::vector<unsigned>&  getSolidSelection() const ;

    std::vector<unsigned> m_solid_selection ; 
};





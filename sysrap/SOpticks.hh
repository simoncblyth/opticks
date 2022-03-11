#pragma once
/**
SOpticks.hh
==============

Collecting the subset of Opticks API needed in CSGOptiX as it 
would be helpful to be able to build CSGOptiXRenderTest with 
a really minimal install excluding : brap, NPY, optickscore




**/

#include <vector>
#include "SYSRAP_API_EXPORT.hh"

struct SArgs ; 
class Composition ; 


struct SYSRAP_API SOpticks
{
    SOpticks(int argc, char** argv, const char* argforced ) ; 

    bool                          hasArg(const char* arg) const ; 
    bool                          isEnabledMergedMesh(unsigned ridx); 
    std::vector<unsigned>&        getSolidSelection() ;
    const std::vector<unsigned>&  getSolidSelection() const ;
    int                           getRaygenMode() const ; 

    Composition*                  getComposition() const ; 
    const char*                   getOutDir() const ;  

    SArgs*                        m_sargs ;
    std::vector<unsigned>         m_solid_selection ; 
};





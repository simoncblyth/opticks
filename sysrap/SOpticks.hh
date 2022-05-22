#pragma once
/**
SOpticks.hh
==============

Collecting the subset of Opticks API needed in CSGOptiX as it 
would be helpful to be able to build CSGOptiXRenderTest with 
a really minimal install excluding : brap, NPY, optickscore

The major difficulty with that is Composition which brings in a large number of classes. 
TODO: review Composition and dependents and see how difficult to relocate to sysrap 

**/

#include <vector>
#include "plog/Severity.h"
#include "SYSRAP_API_EXPORT.hh"

struct SArgs ; 
class Composition ; 


struct SYSRAP_API SOpticks
{
    static const plog::Severity LEVEL ; 

    static const char* CFBaseScriptPath() ; 
    static std::string CFBaseScriptString(const char* cfbase, const char* msg); 
    static void WriteCFBaseScript(const char* cfbase, const char* msg);
    static void WriteOutputDirScript(const char* outdir); 
    static void WriteOutputDirScript(); 

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





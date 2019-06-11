#pragma once

#include <string>
#include <glm/fwd.hpp>
#include "plog/Severity.h"

class GMergedMesh ; 

class Opticks ; 
class OpticksHub ; 
class Composition ; 

#include "OKGEO_API_EXPORT.hh"

/**
OpticksAim
===========

Canonical m_aim is resident of OpticksHub and is instanciated by OpticksHub::init
The crucial for domain setup OpticksAim::registerGeometry is 


**/

class OKGEO_API OpticksAim {
    public:
       static const plog::Severity LEVEL ; 
    public:
       OpticksAim(OpticksHub* hub);
       void registerGeometry(GMergedMesh* mm0);
    public:
       void            target();   // point composition at geocenter or the m_evt (last created)
       void            setTarget(unsigned target=0, bool aim=true);
       void            setupCompositionTargetting() ;
       unsigned        getTarget();
    private:
       glm::vec4       getCenterExtent();
       unsigned        getTargetDeferred();
       void            dumpTarget(const char* msg="OpticksAim::dumpTarget");  
    private:
       Opticks*     m_ok ; 
       bool         m_dbgaim ;  // --dbgaim
       OpticksHub*  m_hub ; 
       Composition* m_composition ; 

       GMergedMesh*         m_mesh0 ; 
       unsigned             m_target ;
       unsigned             m_target_deferred ;
 

};



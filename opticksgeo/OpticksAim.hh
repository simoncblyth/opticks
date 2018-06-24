#pragma once

#include <string>
#include <glm/fwd.hpp>

class GMergedMesh ; 

class Opticks ; 
class OpticksHub ; 
class Composition ; 

#include "OKGEO_API_EXPORT.hh"

/**
OpticksAim
===========



**/

class OKGEO_API OpticksAim {
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
       OpticksHub*  m_hub ; 
       Composition* m_composition ; 

       GMergedMesh*         m_mesh0 ; 
       unsigned             m_target ;
       unsigned             m_target_deferred ;
 

};



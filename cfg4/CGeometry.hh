#pragma once

#include <string>
#include <map>

class OpticksHub ; 
class CG4 ; 
class Opticks ; 
template <typename T> class OpticksCfg ; 
class CDetector ; 
class CMaterialLib ; 
class CMaterialTable ; 
class CMaterialBridge ; 
class CSurfaceBridge ; 
class CSensitiveDetector ; 

#include "CFG4_API_EXPORT.hh"

/**
CGeometry
===========

1. init with CGDMLDetector or CTestDetector when using "--test" option 
2. relies on creator(CG4) to call CGeometry::hookup(CG4* g4) giving the geometry to Geant4 


**/

class CFG4_API CGeometry 
{
   public:
       CGeometry(OpticksHub* hub, CSensitiveDetector* sd);
       bool hookup(CG4* g4);
       void postinitialize();   // invoked by CG4::postinitialize after Geant4 geometry constructed
   public:
       CMaterialLib*    getMaterialLib() const ;
       CDetector*       getDetector() const ;
       CMaterialBridge* getMaterialBridge() const ;
       CSurfaceBridge*  getSurfaceBridge() const ;
       const std::map<std::string, unsigned>& getMaterialMap() const ;        
   private:
       void init();
       void export_();
   private:
       OpticksHub*          m_hub ; 
       CSensitiveDetector*  m_sd ; 

       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       CDetector*           m_detector ; 
       CMaterialLib*        m_mlib ; 
       CMaterialTable*      m_material_table ; 
       CMaterialBridge*     m_material_bridge ; 
       CSurfaceBridge*      m_surface_bridge ; 

};




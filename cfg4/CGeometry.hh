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

#include "CFG4_API_EXPORT.hh"

class CFG4_API CGeometry 
{
   public:
       CGeometry(OpticksHub* hub);
       bool hookup(CG4* g4);
       void postinitialize();   // invoked by CG4::postinitialize after Geant4 geometry constructed
   public:
       CMaterialLib*    getMaterialLib();
       CDetector*       getDetector();
       CMaterialBridge* getMaterialBridge();
       CSurfaceBridge*  getSurfaceBridge();
       std::map<std::string, unsigned>& getMaterialMap();        
   private:
       void init();
       void export_();
   private:
       OpticksHub*          m_hub ; 

       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       CDetector*           m_detector ; 
       CMaterialLib*        m_mlib ; 
       CMaterialTable*      m_material_table ; 
       CMaterialBridge*     m_material_bridge ; 
       CSurfaceBridge*      m_surface_bridge ; 

};




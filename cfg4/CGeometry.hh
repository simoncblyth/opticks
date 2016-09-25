#pragma once

class OpticksHub ; 
class GGeo ; 
class GSurLib ; 
class CG4 ; 
class Opticks ; 
template <typename T> class OpticksCfg ; 
class CDetector ; 
class CPropLib ; 


#include "CFG4_API_EXPORT.hh"

class CFG4_API CGeometry 
{
   public:
       CGeometry(OpticksHub* hub);
       bool hookup(CG4* g4);

       CPropLib*  getPropLib();
       CDetector* getDetector();
   private:
       void init();
       void kludgeSurfaces();
   private:
       OpticksHub*          m_hub ; 
       GGeo*                m_ggeo ; 
       GSurLib*             m_surlib ; 

       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       CDetector*           m_detector ; 
       CPropLib*            m_lib ; 

};




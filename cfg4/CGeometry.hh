#pragma once

class OpticksHub ; 
class CG4 ; 
class Opticks ; 
template <typename T> class OpticksCfg ; 
class CDetector ; 
class CPropLib ; 


#include "CFG4_API_EXPORT.hh"

class CFG4_API CGeometry 
{
   public:
       CGeometry(OpticksHub* hub, CG4* g4);
       CPropLib*  getPropLib();
       CDetector* getDetector();
   private:
       void init();
   private:
       OpticksHub*          m_hub ; 
       CG4*                 m_g4 ; 
       Opticks*             m_ok ; 
       OpticksCfg<Opticks>* m_cfg ; 
       CDetector*           m_detector ; 
       CPropLib*            m_lib ; 

};




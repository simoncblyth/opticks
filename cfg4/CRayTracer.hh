#pragma once

// g4-
class G4VFigureFileMaker ;
class G4VRTScanner ;
class G4TheRayTracer ; 

// okc-
class Opticks ; 
class Composition ; 

// okg-
class OpticksHub ; 

// cfg4-
class CG4 ; 

#include "CFG4_API_EXPORT.hh"

/**
CRayTracer
============

Canonical m_rt instance is ctor resident of CG4. 

**/


class CFG4_API CRayTracer
{
   public:
        CRayTracer(CG4* g4);
        void snap() const ;
   private:
        CG4*         m_g4 ; 
        Opticks*     m_ok ;
        OpticksHub*  m_hub ; 
        Composition* m_composition ; 

        G4VFigureFileMaker* m_figmaker ; 
        G4VRTScanner*       m_scanner ; 
        G4TheRayTracer*     m_tracer ; 
  
};

    

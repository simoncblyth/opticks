#pragma once

class Opticks ; 
class OpticksHub ; 
class CG4 ; 
class G4RunManager ; 

//#define OLDPHYS 1
#ifdef OLDPHYS
class PhysicsList ; 
#else
class OpNovicePhysicsList ; 
#endif

#include "CFG4_API_EXPORT.hh"

class CFG4_API CPhysics 
{
    public:
        CPhysics(CG4* g4);
    public:
        G4RunManager* getRunManager(); 
        void setProcessVerbosity(int verbosity);
    private:
        void init();
    private:
        CG4*           m_g4 ;     
        OpticksHub*    m_hub ;     
        Opticks*       m_ok ;     
        G4RunManager*  m_runManager ; 
#ifdef OLDPHYS
        PhysicsList*          m_physics ; 
#else
        OpNovicePhysicsList*  m_physics ; 
#endif

};




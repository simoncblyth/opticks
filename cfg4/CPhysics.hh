#pragma once

class Opticks ; 
class OpticksHub ; 
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
        CPhysics(OpticksHub* hub);
    public:
        G4RunManager* getRunManager(); 
        void setProcessVerbosity(int verbosity);
    private:
        void init();
    private:
        OpticksHub*    m_hub ;     
        Opticks*       m_ok ;     
        G4RunManager*  m_runManager ; 
#ifdef OLDPHYS
        PhysicsList*          m_physics ; 
#else
        OpNovicePhysicsList*  m_physics ; 
#endif

};




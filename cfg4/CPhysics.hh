#pragma once

class Opticks ; 
class OpticksHub ; 
class CG4 ; 
class G4RunManager ; 

//#define OLDPHYS 1
#ifdef OLDPHYS
class PhysicsList ; 
#else
//class OpNovicePhysicsList ; 
class CPhysicsList ; 
#endif

#include "CFG4_API_EXPORT.hh"

/**
CPhysics
==========

HUH: why the runManager lives here ? , expected CG4

**/

class CFG4_API CPhysics 
{
    public:
        CPhysics(CG4* g4);
    public:
        G4RunManager* getRunManager() const ; 
        void setProcessVerbosity(int verbosity);
    private:
        void init();
    private:
        CG4*           m_g4 ;     
        OpticksHub*    m_hub ;     
        Opticks*       m_ok ;     
        G4RunManager*  m_runManager ; 
#ifdef OLDPHYS
        PhysicsList*          m_physicslist ; 
#else
        //OpNovicePhysicsList*  m_physicslist ; 
        CPhysicsList*           m_physicslist ; 
#endif

};




#pragma once

//class G4MaterialPropertyVector; 
class G4PhysicsOrderedFreeVector ;


#include "CFG4_API_EXPORT.hh"
class CFG4_API CVec 
{
    public:
         CVec(G4PhysicsOrderedFreeVector* vec) ; 
         float getValue(float nm);
         void  dump(const char* msg="CVec::dump", float lo=60.f, float hi=720.f, float step=20.f);
    private:
         G4PhysicsOrderedFreeVector* m_vec ; 

};

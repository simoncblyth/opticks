#pragma once

#include <string>
//class G4MaterialPropertyVector; 
class G4PhysicsOrderedFreeVector ;


#include "CFG4_API_EXPORT.hh"
class CFG4_API CVec 
{
    public: 
         static std::string Digest(CVec* vec);  
         static std::string Digest(G4PhysicsOrderedFreeVector* vec);  // see G4PhysicsOrderedFreeVectorTest
         static CVec* MakeDummy(size_t n ); 
    public:
         CVec(G4PhysicsOrderedFreeVector* vec) ; 
         std::string  digest();
         G4PhysicsOrderedFreeVector* getVec(); 
         float getValue(float nm);
         void  dump(const char* msg="CVec::dump", float lo=60.f, float hi=720.f, float step=20.f);
    private:
         G4PhysicsOrderedFreeVector* m_vec ; 

};

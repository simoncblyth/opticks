#pragma once

#include <map>
#include <string>
class CG4 ; 
struct CG4Ctx ; 
class CMaterialLib ; 

//#include "G4ThreeVector.hh"
#include "G4Transportation.hh"

class DebugG4Transportation : public G4Transportation 
{
   public:
       DebugG4Transportation( CG4* g4, G4int verbosityLevel= 1); 
   private:
       void init();
   public:
       std::string firstMaterialWithGroupvelAt430nm(float groupvel, float delta=0.001f);
       G4VParticleChange* AlongStepDoIt( const G4Track& track, const G4Step&  stepData );

   private:
       CG4*          m_g4 ; 
       CG4Ctx&       m_ctx ; 
       CMaterialLib* m_mlib ; 
       G4ThreeVector m_origin ; 


};





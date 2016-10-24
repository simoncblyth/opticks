#pragma once

#include <string>
#include "G4ThreeVector.hh" 

class G4Step ; 

#include "CFG4_API_EXPORT.hh"

class CFG4_API CStp 
{
   public:
        CStp(const G4Step* step, int step_id, const G4ThreeVector& origin) ;
        std::string description();
   private:
         const G4Step* m_step ; 
         int           m_step_id ; 
         G4ThreeVector m_origin ;

};


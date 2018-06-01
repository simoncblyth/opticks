#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class Opticks ;
class G4VPhysicalVolume ; 

class CFG4_API CImporter {
   public: 
       CImporter(Opticks* ok, G4VPhysicalVolume* top);
   private:
       Opticks*            m_ok ; 
       G4VPhysicalVolume* m_top ; 

};

#include "CFG4_TAIL.hh"

    

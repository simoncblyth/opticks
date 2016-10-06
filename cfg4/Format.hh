#pragma once

#include <string>
#include <vector>

class G4Track ; 
class G4Step ; 
class G4StepPoint ; 
class CStep ; 

#include "G4ThreeVector.hh"
#include "CFG4_API_EXPORT.hh"


CFG4_API std::string Format(const G4Track* track,  const G4ThreeVector& origin, const char* msg="Track", bool op=true );
CFG4_API std::string Format(const G4Step* step,    const G4ThreeVector& origin, const char* msg="Step",  bool op=true );
CFG4_API std::string Format(const G4StepPoint* sp, const G4ThreeVector& origin, const char* msg="Pt",    bool op=true );

CFG4_API std::string Format(const G4ThreeVector& vec, const char* msg="Vec", unsigned int fwid=4);
CFG4_API std::string Format(std::vector<const CStep*>& steps, const char* msg, bool op=true );

CFG4_API std::string Format(const char* label, std::string pre, std::string post, unsigned int w=20);

CFG4_API std::string Tail(const G4String& s, unsigned int n );


 

#pragma once

#include <glm/fwd.hpp>
#include <string>

#include "OpticksCSG.h"


class NCSG ; 
struct nnode ; 

class GCSG ; 
class G4VSolid;
class Opticks ; 


/**
CMaker
======

CMaker is a constitent of CTestDetector used
to convert GCSG geometry into G4 geometry in 
G4VPhysicalVolume* CTestDetector::Construct(). 

CMaker::makeSolid handles some boolean intersection
and union combinations via recursive calls to itself.

CMaker only handles the geometrical shapes.
Material assignments are done elsewhere, 
at a higher level eg by CTestDetector.

**/

#include "CFG4_API_EXPORT.hh"

class CFG4_API CMaker 
{
    public:
        static std::string PVName(const char* shapename);
        static std::string LVName(const char* shapename);
    public:
        CMaker(Opticks* ok, int verbosity=0);
    public:
        G4VSolid* makeSolid(OpticksCSG_t type, const glm::vec4& param);
        G4VSolid* makeBox(const glm::vec4& param);
        G4VSolid* makeSphere(const glm::vec4& param);
    public:
        G4VSolid* makeSolid(GCSG* csg, unsigned int i);  // ancient CSG 
    public:
        // current CSG 
        G4VSolid* makeSolid(NCSG* csg);                  
    private:
        G4VSolid* makeSolid_r(const nnode* node);
    private:
        Opticks* m_ok ; 
        int      m_verbosity ; 

};



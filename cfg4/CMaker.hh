#pragma once

#include <glm/fwd.hpp>
#include <string>

#include "OpticksCSG.h"
#include <glm/fwd.hpp>

class NCSG ; 
struct nnode ; 

class GCSG ; 
class G4VSolid;
class Opticks ; 

namespace HepGeom
{
   class Transform3D ; 
}

typedef HepGeom::Transform3D G4Transform3D;

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
        static std::string PVName(const char* shapename, int idx=-1);
        static std::string LVName(const char* shapename, int idx=-1);
    public:
        CMaker(Opticks* ok, int verbosity=0);
    public:
        // primordial CSG
        G4VSolid* makeSolid_OLD(OpticksCSG_t type, const glm::vec4& param);
        G4VSolid* makeBox_OLD(const glm::vec4& param);
        G4VSolid* makeSphere_OLD(const glm::vec4& param);
        G4VSolid* makeSolid_OLD(GCSG* csg, unsigned int i); 
    public:
        // current CSG 
        G4VSolid* makeSolid(const NCSG* csg);                  
    private:
        G4VSolid* makeSolid_r(const nnode* node);
        static G4Transform3D* ConvertTransform(const glm::mat4& t);
        static G4VSolid*      ConvertPrimitive(const nnode* node);
    private:
        Opticks* m_ok ; 
        int      m_verbosity ; 

};



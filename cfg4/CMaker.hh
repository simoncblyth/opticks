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
to convert NCSG/nnode shapes into G4VSolid. 
CMaker::makeSolid handles booleans via recursive calls.

An earlier implementation from prior to NCSG/nnode is in CMaker.cc.old 


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
        G4VSolid* makeSolid(const NCSG* csg);                  
    private:
        G4VSolid* makeSolid_r(const nnode* node);
        static G4Transform3D* ConvertTransform(const glm::mat4& t);
        static G4VSolid*      ConvertPrimitive(const nnode* node);
        static G4VSolid*      ConvertConvexPolyhedron(const nnode* node);
    private:
        Opticks* m_ok ; 
        int      m_verbosity ; 

};



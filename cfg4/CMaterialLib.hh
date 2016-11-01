#pragma once

#include <map>
#include <string>

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class OpticksHub ; 

class GMaterial ; 
class G4Material ; 

#include "CPropLib.hh"

/**
CMaterialLib
===============

CMaterialLib is a constituent of CDetector (eg CTestDector and CGDMLDetector)
that converts GGeo (ie Opticks G4DAE) materials and surfaces into G4 materials and surfaces.
The GGeo gets loaded on initializing base class CPropLib.

**/


class CFG4_API CMaterialLib : public CPropLib 
{
   public:
       CMaterialLib(OpticksHub* hub);

       void convert(); // commented in init 
       const G4Material* makeInnerMaterial(const char* spec);  // TODO: CMaterialLib better to not know about bnd spec
       const G4Material* makeMaterial(const char* matname);

       void dump(const char* msg="CMaterialLib::dump");

       // G4 material access
       const G4Material* getG4Material(const char* shortname);
   private:
       void dump(const GMaterial* mat, const char* msg="CMaterialLib::dump");
       void dumpMaterials(const char* msg="CMaterialLib::dumpMaterials");
       void dumpMaterial(const G4Material* mat, const char* msg="CMaterialLib::dumpMaterial");

   private:
       const G4Material* convertMaterial(const GMaterial* kmat);
       G4Material* makeVacuum(const char* name);
       G4Material* makeWater(const char* name);

   private:
       bool                                            m_converted ;      
       std::map<const GMaterial*, const G4Material*>   m_ggtog4 ; 
       std::map<std::string, const G4Material*>        m_g4mat ; 



};

#include "CFG4_TAIL.hh"


#pragma once

#include <map>
#include <string>

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

template <typename T> class NPY ; 

class OpticksHub ; 

class GMaterial ; 
class G4Material ; 

class CMPT ; 
#include "CPropLib.hh"

/**
CMaterialLib
===============

CMaterialLib is a constituent of CDetector (eg CTestDector and CGDMLDetector)
that converts GGeo (ie Opticks G4DAE) materials and surfaces into G4 materials and surfaces.
G4The GGeo gets loaded on initializing base class CPropLib.

WAS SURPRISED TO FIND THAT THE CONVERSION IS NOT DONE BY STANDARD LAUNCH
INSTEAD THE INDIVIDUAL convertMaterial ARE CALLED FROM EG CTestDetector 
WHICH POPULATES THE MAP.

**/


class CFG4_API CMaterialLib : public CPropLib 
{
   public:
       CMaterialLib(OpticksHub* hub);

       bool isConverted();

       void convert(); // commented in init, never invoked in standard running 
       void postinitialize();  // invoked from CGeometry::postinitialize 

       const G4Material* makeG4Material(const char* matname);
       const G4Material* convertMaterial(const GMaterial* kmat);

       void dump(const char* msg="CMaterialLib::dump");
       void saveGROUPVEL(const char* base="$TMP/CMaterialLib");

       // G4 material access
       bool hasG4Material(const char* shortname);
       const G4Material* getG4Material(const char* shortname);
       const CMPT*       getG4MPT(const char* shortname);


       void dumpGroupvelMaterial(const char* msg, float wavelength, float groupvel, float tdiff, int step_id, const char* qwn="" );
       std::string firstMaterialWithGroupvelAt430nm(float groupvel, float delta=0.0001f);
       void fillMaterialValueMap();
       void fillMaterialValueMap(std::map<std::string,float>& vmp,  const char* matnames, const char* key, float nm);
       static void dumpMaterialValueMap(const char* msg, std::map<std::string,float>& vmp);
       static std::string firstKeyForValue(float val, std::map<std::string,float>& vmp, float delta=0.0001f );
       


       const G4Material* getG4Material(unsigned index);
       NPY<float>* makeArray(const char* name, const char* keys, bool reverse=true);
   private:
       void dump(const GMaterial* mat, const char* msg="CMaterialLib::dump");
       void dumpMaterials(const char* msg="CMaterialLib::dumpMaterials");
       void dumpMaterial(const G4Material* mat, const char* msg="CMaterialLib::dumpMaterial");

   private:
       G4Material* makeVacuum(const char* name);
       G4Material* makeWater(const char* name);

   private:
       bool                                            m_converted ;      
       std::map<const GMaterial*, const G4Material*>   m_ggtog4 ; 
       std::map<std::string, const G4Material*>        m_g4mat ; 
   private:
       std::map<std::string, float>                    m_groupvel_430nm ; 


};

#include "CFG4_TAIL.hh"


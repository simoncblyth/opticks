#pragma once

#include <cstddef>
#include <map>

// ggeo-
class GCache ; 
class GBndLib ;
class GMaterialLib ;
class GSurfaceLib ;

class GCSG ; 
class GMaterial ;

// npy-
struct NSlice ; 

// g4-
class G4Material ; 
class G4MaterialPropertiesTable ; 


class CPropLib {
   public:
       CPropLib(GCache* cache);
   private:
       void init();
   public:
       G4Material* makeInnerMaterial(const char* spec);
       G4Material* makeMaterial(const char* matname);
       GCSG*       getPmtCSG(NSlice* slice);
       unsigned int getMaterialIndex(G4Material* material);
   public:
       void dumpMaterials(const char* msg="CPropLib::dumpMaterials");
   private:
       G4Material* convertMaterial(GMaterial* kmat);
       G4Material* makeVacuum(const char* name);
       G4Material* makeWater(const char* name);
       G4MaterialPropertiesTable* makeMaterialPropertiesTable(GMaterial* kmat);
   private:
       GCache*            m_cache ; 
       GBndLib*           m_bndlib ; 
       GMaterialLib*      m_mlib ; 
       GSurfaceLib*       m_slib ; 

       std::map<GMaterial*, G4Material*>   m_ggtog4 ; 
       std::map<G4Material*, unsigned int> m_g4toix ; 

};

inline CPropLib::CPropLib(GCache* cache)
  : 
  m_cache(cache),
  m_bndlib(NULL),
  m_mlib(NULL),
  m_slib(NULL)
{
    init();
}




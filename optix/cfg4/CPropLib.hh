#pragma once

#include <cstddef>

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



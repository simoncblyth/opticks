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

template <typename T> class GProperty ; 
template <typename T> class GPropertyMap ; 

// npy-
struct NSlice ; 

// g4-
class G4Material ; 
class G4MaterialPropertiesTable ; 


class CPropLib {
   public:
       static const char* SENSOR_MATERIAL ;
   public:
       CPropLib(GCache* cache, int verbosity=0);
   private:
       void init();
   public:
       const G4Material* makeInnerMaterial(const char* spec);
       const G4Material* makeMaterial(const char* matname);
       GCSG*       getPmtCSG(NSlice* slice);
       unsigned int getMaterialIndex(const G4Material* material);
   public:
       void dumpMaterials(const char* msg="CPropLib::dumpMaterials");
   private:
       const G4Material* convertMaterial(const GMaterial* kmat);
       G4Material* makeVacuum(const char* name);
       G4Material* makeWater(const char* name);
       G4MaterialPropertiesTable* makeMaterialPropertiesTable(const GMaterial* kmat);
       void addProperty(G4MaterialPropertiesTable* mpt, const char* lkey,  GProperty<float>* prop );
   private:
       GCache*            m_cache ; 
       int                m_verbosity ; 
       GBndLib*           m_bndlib ; 
       GMaterialLib*      m_mlib ; 
       GSurfaceLib*       m_slib ; 
       GPropertyMap<float>* m_sensor_surface ; 

       std::map<const GMaterial*, const G4Material*>   m_ggtog4 ; 
       std::map<const G4Material*, unsigned int> m_g4toix ; 

};

inline CPropLib::CPropLib(GCache* cache, int verbosity)
  : 
  m_cache(cache),
  m_verbosity(verbosity),
  m_bndlib(NULL),
  m_mlib(NULL),
  m_slib(NULL)
{
    init();
}




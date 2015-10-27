#pragma once

#include <vector>
#include "GPropertyLib.hh"

class GMaterial ; 
class GItemList ; 

//
// rationale: 
//    * need dynamic boundary construction post-cache
//
// approach: 
//    * constituent of GGeo that collects materials, standardizes and persists to NPY<float>
//    * take over GMaterial handling from GGeo and GBoundaryLib
//    * aiming togther with a GSurfaceLib to greatly simplify GBoundaryLib 
//

class GMaterialLib : public GPropertyLib {
   public:
       // 4 standard material property names : interleaved into float4 wavelength texture
       static const char* propertyName(unsigned int k);
       static const char* refractive_index ; 
       static const char* absorption_length ; 
       static const char* scattering_length ; 
       static const char* reemission_prob ; 
   public:
       static const char* keyspec ;
   public:
       static GMaterialLib* load(GCache* cache);
       GMaterialLib(GCache* cache); 
       void Summary(const char* msg="GMaterialLib::Summary");
       void dump(const char* msg="GMaterialLib::dump");
       void dump(GMaterial* mat, const char* msg="GMaterialLib::dump");
   private:
       void init();
   public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       void import();
       NPY<float>* createBuffer();
       GItemList*  createNames();
   public:
       // lifecycle
       void add(GMaterial* material);
   public:
       GMaterial* getMaterial(unsigned int i);
       unsigned int getNumRawMaterials();
       unsigned int getNumMaterials();
   private:
       GMaterial*  createStandardMaterial(GMaterial* src);
   private:
       // post-cache
       void import( GMaterial* mat, float* data, unsigned int nj, unsigned int nk );
   private:
       std::vector<GMaterial*>       m_materials_raw ; 
       std::vector<GMaterial*>       m_materials ; 


};

inline GMaterialLib::GMaterialLib(GCache* cache) 
    :
    GPropertyLib(cache, "GMaterialLib")
{
    init();
}
 
inline unsigned int GMaterialLib::getNumMaterials()
{
    return m_materials.size();
}
inline unsigned int GMaterialLib::getNumRawMaterials()
{
    return m_materials_raw.size();
}
inline GMaterial* GMaterialLib::getMaterial(unsigned int i)
{
    return m_materials[i] ;
}

 

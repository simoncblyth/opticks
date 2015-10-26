#pragma once

#include <vector>
#include "GPropertyLib.hh"

class GMaterial ; 

//
// rationale: 
//    * need dynamic boundary construction post-cache
//
// approach: 
//    * constituent of GGeo that collects materials, standardizes and persists to GBuffer
//    * take over GMaterial handling from GGeo and GBoundaryLib
//    * aiming togther with a GSurfaceLib to greatly simplify GBoundaryLib 
//

class GMaterialLib : public GPropertyLib {
   public:
      // 4 standard material property names : interleaved into float4 wavelength texture
      static const char* refractive_index ; 
      static const char* absorption_length ; 
      static const char* scattering_length ; 
      static const char* reemission_prob ; 
   public:
      static const char* keyspec ;
   public:
       GMaterialLib(GCache* cache); 
       void Summary(const char* msg="GMaterialLib::Summary");
   private:
       void init();
   public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
   public:
       void add(GMaterial* material);
       void addRaw(GMaterial* material);
   public:
       unsigned int getNumMaterials();
       unsigned int getNumRawMaterials();
   private:
       std::vector<GMaterial*>       m_materials ; 
       std::vector<GMaterial*>       m_materials_raw ; 

};

inline GMaterialLib::GMaterialLib(GCache* cache) 
    :
    GPropertyLib(cache)
{
    init();
}
 
inline void GMaterialLib::add(GMaterial* material)
{
    m_materials.push_back(material);
}
inline void GMaterialLib::addRaw(GMaterial* material)
{
    m_materials_raw.push_back(material);
}
inline unsigned int GMaterialLib::getNumMaterials()
{
    return m_materials.size();
}
inline unsigned int GMaterialLib::getNumRawMaterials()
{
    return m_materials_raw.size();
}
 

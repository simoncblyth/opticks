#pragma once
/*
GMaterialLib
=============

Objective is to enable deferred boundary construction post-cache by persisting 
Materials and Surfaces rather than directly persisting Boundaries.

Contrary to initial thinking *GMaterialLib* now handles
**only standardized material** collection. 
General raw material handling is still done directly by *GGeo*.
 
The *GPropertyLib* subclasses aim to replace *GBoundaryLib*.

* *GMaterialLib* 
* *GSurfaceLib*
* *GScintillatorLib*
* *GBndLib* 

Lifecycle of all property lib are similar:

*ctor*
     constituent of GGeo instanciated in GGeo::init when running precache 
     or via GGeo::loadFromCache when running from cache

*init*
     invoked by *ctor*, sets up the keymapping and default properties 
     that are housed in GPropertyLib base

*add*
     from GGeo::loadFromG4DAE (ie in precache running only) 
     GMaterial instances are collected via AssimpGGeo::convertMaterials and GGeo::add

*close*
     GPropertyLib::close first invokes *sort* and then 
     serializes collected and potentially reordered objects via *createBuffer* 
     and *createNames* 

     * *close* is triggered by the first call to getIndex
     * after *close* no new materials can be added
     * *close* is canonically invoked by GBndLib::getOrCreate during AssimpGGeo::convertStructureVisit 
 
*save*
     buffer and names are written to cache by GPropertyLib::saveToCache

*load*
     static method that instanciates and populates via GPropertyLib::loadFromCache which
     reads in the buffer and names and then invokes *import*
     This allows operation from the cache without having to GGeo::loadFromG4DAE.

*import*
     reconstitutes the serialized objects and populates the collection of them
     TODO: digest checking the reconstitution

*/
#include <vector>
#include "GPropertyLib.hh"

class GMaterial ; 
class GItemList ; 

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
       void save();
       static GMaterialLib* load(GCache* cache);
   public:
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
       void sort();
       bool operator()(const GMaterial& a_, const GMaterial& b_);
   public:
       GMaterial* getMaterial(unsigned int i); // zero based index
       unsigned int getNumMaterials();
   private:
       GMaterial*  createStandardMaterial(GMaterial* src);
   private:
       // post-cache
       void import( GMaterial* mat, float* data, unsigned int nj, unsigned int nk );
   private:
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
inline GMaterial* GMaterialLib::getMaterial(unsigned int i)
{
    return m_materials[i] ;
}

 

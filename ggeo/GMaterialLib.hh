#pragma once
/*
GMaterialLib
=============

Objective is to enable deferred boundary construction post-cache by persisting 
Materials and Surfaces rather than directly persisting Boundaries.

Contrary to initial thinking *GMaterialLib* now handles
**only standardized material** collection. 
General raw material handling is still done directly by *GGeo*.
 
The below *GPropertyLib* subclasses replace the former *GBoundaryLib*.

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

class GMaterial ; 
class GItemList ; 

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GMaterialLib : public GPropertyLib {
   public:
       // 4 standard material property names : interleaved into float4 wavelength texture
       static const float MATERIAL_UNSET ; 
       static const char* propertyName(unsigned int k);
   public:
       static const char* refractive_index ; 
       static const char* absorption_length ; 
       static const char* scattering_length ; 
       static const char* reemission_prob ; 
   public:
       static const char* group_velocity ; 
       static const char* extra_y ; 
       static const char* extra_z ; 
       static const char* extra_w ; 
   public:
       static const char* refractive_index_local ; 
   public:
       static const char* keyspec ;
   public:
       void save();
       static GMaterialLib* load(Opticks* cache);
   public:
       GMaterialLib(Opticks* cache); 
   public:
       void Summary(const char* msg="GMaterialLib::Summary");
       void dump(const char* msg="GMaterialLib::dump");
       void dump(GMaterial* mat, const char* msg);
       void dump(GMaterial* mat);
       void dump(unsigned int index);
   private:
       void init();
   public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       void import();
       NPY<float>* createBuffer();
       GItemList*  createNames();
   private:
       void importForTex2d();
       void importOld();
       NPY<float>* createBufferForTex2d();
       NPY<float>* createBufferOld();
   public:
       // lifecycle
       void add(GMaterial* material);
       void sort();
       bool operator()(const GMaterial& a_, const GMaterial& b_);
   public:
       void addTestMaterials();
   public:
       bool hasMaterial(unsigned int index);
       bool hasMaterial(const char* name);
       GMaterial* getMaterial(const char* name); 
       // base class provides: unsigned getIndex(const char* shortname)
       GMaterial* getMaterial(unsigned int i); // zero based index
       const char* getNameCheck(unsigned int i);
       unsigned int getNumMaterials();
       unsigned int getMaterialIndex(const GMaterial* material);
   public:
       GMaterial*  createStandardMaterial(GMaterial* src);
   private:
       // post-cache
       void import( GMaterial* mat, float* data, unsigned int nj, unsigned int nk, unsigned int jcat=0 );
   private:
       std::vector<GMaterial*>       m_materials ; 

};
#include "GGEO_TAIL.hh"

 

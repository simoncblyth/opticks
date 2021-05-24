/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#pragma once
/*
GMaterialLib
=============

Objective is to enable deferred boundary construction post-cache by persisting 
Materials and Surfaces rather than directly persisting Boundaries.

Contrary to initial thinking *GMaterialLib* now handles
**only standardized material** collection. 
General raw material handling is still done directly by *GGeo*.


Material Ordering 
------------------

Ordering is mostly an old approach possibility, 
in the new live from G4 approach there is no detector 
identification : so no opportunity for per detector preference
order.  In the new approach change order at source by changing 
the order of instanciation of the G4Material.


Cathode material
------------------

Currently a single material with a non-zero EFFICIENCY value will
result in the "Cathode" material being set.  This is used by 
GGeoSensor::AddSensorSurfaces which gets invoked 
by X4PhysicalVolume::convertSensors in the direct geometry workflow. 
It results in the addition of "technical" GSkinSurface to the geometry 





*/
#include <vector>

class BMeta ; 
class GMaterial ; 
class GItemList ; 

#include "plog/Severity.h"
#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GMaterialLib : public GPropertyLib {
   public:
       static const plog::Severity  LEVEL ;
       static const GMaterialLib* INSTANCE ; 
       static const GMaterialLib* GetInstance() ; 
       static bool  IsUnset(unsigned index); 
   public:
       friend class X4PhysicalVolume ; 
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
       typedef enum { ORDER_ASIS, ORDER_BY_SRCIDX, ORDER_BY_PREFERENCE } MaterialOrder_t ; 
       static const char* ORDER_ASIS_  ; 
       static const char* ORDER_BY_SRCIDX_  ; 
       static const char* ORDER_BY_PREFERENCE_ ; 
       const char* getMaterialOrdering() const ;
   public:
       void save();
       static GMaterialLib* load(Opticks* cache);
   public:
       GMaterialLib(Opticks* ok, GMaterialLib* basis=NULL); 
   public:
       GMaterialLib(GMaterialLib* other, GDomain<float>* domain=NULL, GMaterialLib* basis=NULL);  // interpolating copy ctor
   public:
       static void dump(GMaterial* mat, const char* msg);
       static void dump(GMaterial* mat);
   public:
       void Summary(const char* msg="GMaterialLib::Summary");
       void dump(const char* msg="GMaterialLib::dump");
       void dump(unsigned int index);
   private:
       void init();
       void initInterpolatingCopy(GMaterialLib* src, GDomain<float>* domain);
   public:
       // concretization of GPropertyLib
       void defineDefaults(GPropertyMap<float>* defaults); 
       void import();
       void beforeClose(); 
       void postLoadFromCache();
       bool setMaterialPropertyValues(const char* matname, const char* propname, float val); // post-import modification

       NPY<float>* createBuffer();
       BMeta*      createMeta();
       GItemList*  createNames();
   private:
       void replaceGROUPVEL(bool debug=false);  // triggered in postLoadFromCache with --groupvel option
       void importForTex2d();
       NPY<float>* createBufferForTex2d();
   public:
       // lifecycle
       void add(GMaterial* material);
       void addRaw(GMaterial* material);
       void addDirect(GMaterial* material);  // not-standarized
       void sort();
       bool order_by_preference(const GMaterial& a_, const GMaterial& b_);
       bool order_by_srcidx(    const GMaterial& a_, const GMaterial& b_);
       bool operator()(const GMaterial& a_, const GMaterial& b_);
#ifdef OLD_CATHODE
    public:
        void setCathode(GMaterial* cathode);
        GMaterial* getCathode() const ;  
        const char* getCathodeMaterialName() const ;
#endif
    private:
        void addSensitiveMaterial(GMaterial* cathode);
    public:
        unsigned getNumSensitiveMaterials() const ;  
        GMaterial* getSensitiveMaterial(unsigned index) const ;   
        void dumpSensitiveMaterials(const char* msg="GMaterialLib::dumpSensitiveMaterials") const ;
   public:
       // used by GGeoTest 
       GMaterial* getBasisMaterial(const char* name) const ;
       void reuseBasisMaterial(const char* name) ;
   public:
       void addTestMaterials();
   public:
       bool hasMaterial(unsigned int index) const ;
       bool hasMaterial(const char* name) const ;
       GMaterial* getMaterial(const char* name) const ; 

       // base class provides: unsigned getIndex(const char* shortname)
       // but that triggers a close 

       GMaterial* getMaterial(unsigned i) const ; // zero based index
       GMaterial* getMaterialWithIndex(unsigned aindex) const ;  
       unsigned  getMaterialIndex(const char* name) const ;    // without triggering a close
       GPropertyMap<float>* findMaterial(const char* shortname) const ; 
       GPropertyMap<float>* findRawMaterial(const char* shortname) const ; 
       GProperty<float>* findRawMaterialProperty(const char* shortname, const char* propname) const ;
       void dumpRawMaterialProperties(const char* msg) const ;
       std::vector<GMaterial*> getRawMaterialsWithProperties(const char* props, char delim) const ;
   public:
       const char* getNameCheck(unsigned int i);
       unsigned getNumMaterials() const ;
       unsigned getNumRawMaterials() const ;
       unsigned int getMaterialIndex(const GMaterial* material);
   public:
       GMaterial*  createStandardMaterial(GMaterial* src);
   private:
       GMaterial*  makeRaw(const char* name);
   private:
       // post-cache
       void import( GMaterial* mat, float* data, unsigned int nj, unsigned int nk, unsigned int jcat=0 );
   private:
       std::vector<GMaterial*>       m_materials ; 
       std::vector<GMaterial*>       m_materials_raw ; 

       GMaterialLib*   m_basis ; 

#ifdef OLD_CATHODE
       GMaterial*      m_cathode ; 
       const char*     m_cathode_material_name ; 
#endif

       std::vector<GMaterial*> m_sensitive_materials ;


       MaterialOrder_t m_material_order ; 


};
#include "GGEO_TAIL.hh"

 

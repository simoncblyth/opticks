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

#include <vector>

/** 
GSurfaceLib
==============

Skin and Border surfaces have an associated optical surface 
that is lodged inside GPropertyMap
in addition to 1(for skin) or 2(for border) volume names

* huh : where are these names persisted ?
    

ISSUE
-------

* domain not persisted, so have to just assume that are using 
  standard one at set on load ?



**/

class BMeta ; 
struct guint4 ; 
class GOpticalSurface ; 
class GSkinSurface ; 
class GBorderSurface ; 
class GItemList ; 
template<typename T> class GProperty ; 

#include "GPropertyLib.hh"
#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"
#include "plog/Severity.h"


class GGEO_API GSurfaceLib : public GPropertyLib {
   public:
       static const plog::Severity LEVEL ; 
       static const char* propertyName(unsigned int k);
       // 4 standard surface property names : interleaved into float4 wavelength texture
  public:
       static const char* detect ;
       static const char* absorb ;
       static const char* reflect_specular ;
       static const char* reflect_diffuse ;
  public:
       static const char* extra_x ; 
       static const char* extra_y ; 
       static const char* extra_z ; 
       static const char* extra_w ; 
  public:
       static const char* AssignSurfaceType( BMeta* surfmeta );
       static const char* BORDERSURFACE ;  
       static const char* SKINSURFACE ;  
       static const char* TESTSURFACE ;  
  public:
       static const char* BPV1 ;  
       static const char* BPV2 ;  
       static const char* SSLV ;  
  public:
       // some model-mismatch translation required for surface properties
       static const char* EFFICIENCY ; 
       static const char* REFLECTIVITY ; 
   public:
       static bool NameEndsWithSensorSurface(const char* name);
       static const char* NameWithoutSensorSurface(const char* name);

       static const char*  SENSOR_SURFACE ;
       static float        SURFACE_UNSET ; 
       static const char* keyspec ;
   public:
       void save();
       static GSurfaceLib* load(Opticks* ok);
   public:
       GSurfaceLib(Opticks* ok, GSurfaceLib* basis=NULL); 
       GSurfaceLib(GSurfaceLib* other, GDomain<float>* domain=NULL, GSurfaceLib* basis=NULL );  // interpolating copy ctor
   private:
       void init();
       void initInterpolatingCopy(GSurfaceLib* src, GDomain<float>* domain);

    public:
        //  
        //  Primary API for populating GSurfaceLib
        //
        void     add(GSkinSurface* ss);
        void     add(GBorderSurface* bs, bool implicit=false, bool direct=false );

   private:
        // methods to assist with de-conflation of surface props and location
        void                  addBorderSurface(GPropertyMap<float>* surf, const char* pv1, const char* pv2, bool direct );
        void                  addSkinSurface(  GPropertyMap<float>* surf, const char* sslv_, bool direct );

        void                  addStandardized(GPropertyMap<float>* surf);
        GPropertyMap<float>*  createStandardSurface(GPropertyMap<float>* src);
        void                  addDirect(GPropertyMap<float>* surf);

        guint4                createOpticalSurface(GPropertyMap<float>* src);
        bool                  checkSurface( GPropertyMap<float>* surf);


   public:
       void Summary(const char* msg="GSurfaceLib::Summary");
       void dump(const char* msg="GSurfaceLib::dump");
       void dumpSurfaces(const char* msg="GSurfaceLib::dumpSurfaces");

       void dumpImplicitBorderSurfaces(const char* msg="GSurfaceLib::dumpImplicitBorderSurfaces") const ; 
       std::string descImplicitBorderSurfaces() const ; 


       void dump(GPropertyMap<float>* surf);
       void dump(GPropertyMap<float>* surf, const char* msg);
       void dump(unsigned int index);
       std::string desc() const ; 
   public:
       void collectSensorIndices();
   public:
       // concretization of GPropertyLib
       void        defineDefaults(GPropertyMap<float>* defaults); 
       NPY<float>* createBuffer();
       BMeta*      createMeta();
       GItemList*  createNames();
   public:
       NPY<float>* createBufferForTex2d();
       NPY<float>* createBufferOld();
   public:
      // methods for debug
       void setFakeEfficiency(float fake_efficiency);
       void addPerfectSurfaces();

       GPropertyMap<float>* makePerfect(const char* name, float detect_, float absorb_, float reflect_specular_, float reflect_diffuse_);
    private: 
       void                 addPerfectProperties( GPropertyMap<float>* dst, float detect_, float absorb_, float reflect_specular_, float reflect_diffuse_ );

   public:
       GPropertyMap<float>* makeImplicitBorderSurface_RINDEX_NoRINDEX(const char* name, const char* pv1, const char* pv2 ) ;

   public:
        GSurfaceLib* getBasis() const ;
        void         setBasis(GSurfaceLib* basis);

        // used from GGeoTest 
        GPropertyMap<float>* getBasisSurface(const char* name) const ; 
        void relocateBasisBorderSurface(const char* name, const char* bpv1, const char* bpv2);
        void relocateBasisSkinSurface(const char* name, const char* sslv);

   public:
       void sort();
       bool operator()(const GPropertyMap<float>* a_, const GPropertyMap<float>* b_);
   public:
       guint4               getOpticalSurface(unsigned int index);  // zero based index
       GPropertyMap<float>* getSensorSurface(unsigned int offset=0);  // 0: first, 1:second 
       unsigned             getNumSensorSurface() const ; 
   public:
       // Check for a surface of specified name of index in m_surfaces vector
       // NB: changed behaviour, formerly named access only worked after closing
       // the lib as used the names buffer     
       GPropertyMap<float>* getSurface(unsigned int index) const ;         // zero based index
       GPropertyMap<float>* getSurface(const char* name) const ;        
       bool                 hasSurface(unsigned int index) const ; 
       bool                 hasSurface(const char* name) const ; 
       GProperty<float>*    getSurfaceProperty(const char* name, const char* prop) const ;


   public:
      // unlike former GBoundaryLib optical buffer one this is surface only
       NPY<unsigned int>* createOpticalBuffer();  
       void               importOpticalBuffer(NPY<unsigned int>* ibuf);
       void               saveOpticalBuffer();
       void               loadOpticalBuffer();
       void               setOpticalBuffer(NPY<unsigned int>* ibuf);
       NPY<unsigned int>* getOpticalBuffer();
   public:
       bool               isSensorSurface(unsigned int surface) const ; // name suffix based, see AssimpGGeo::convertSensor
   public:
       void               import();
   private:
       void dumpMeta(const char* msg="GSurfaceLib::dumpMeta") const ;
       void importOld();
       void importForTex2d();
       void import( GPropertyMap<float>* surf, float* data, unsigned int nj, unsigned int nk, unsigned int jcat=0 );

   public:
       unsigned            getNumSurfaces() const ;  // m_surfaces
       unsigned            getNumBorderSurfaces() const ;
       unsigned            getNumSkinSurfaces() const ;
       GSkinSurface*       getSkinSurface(unsigned index) const ;
       GBorderSurface*     getBorderSurface(unsigned index) const ;

       void                addRaw(GBorderSurface* surface); 
       void                addRaw(GSkinSurface* surface);

       unsigned            getNumRawBorderSurfaces() const ;
       unsigned            getNumRawSkinSurfaces() const ;

       GSkinSurface*       findSkinSurface(const char* lv) const ;
       void                dumpSkinSurface(const char* msg="GSurfaceLib::dumpSkinSurface") const ;

       GBorderSurface*     findBorderSurface(const char* pv1, const char* pv2) const ;
       void                dumpRawSkinSurface(const char* name) const ;
       void                dumpRawBorderSurface(const char* name) const ;

       void                addImplicitBorderSurface_RINDEX_NoRINDEX( const char* pv1, const char* pv2 ); 

   private:
       // primary vector of standardized surfaces, info from which is destined for the GPU texture
       std::vector<GPropertyMap<float>*>       m_surfaces ; 

   private:
       // vectors of non-standardized "input" differentiated surfaces
       std::vector<GSkinSurface*>              m_skin_surfaces ; 
       std::vector<GSkinSurface*>              m_sensor_skin_surfaces ; 
       std::vector<GBorderSurface*>            m_border_surfaces ; 
       std::vector<GBorderSurface*>            m_implicit_border_surfaces ; 

   private:
       // _raw mainly for debug
       std::vector<GSkinSurface*>              m_skin_surfaces_raw ; 
       std::vector<GBorderSurface*>            m_border_surfaces_raw ; 

   private:
       float                                   m_fake_efficiency ; 
       NPY<unsigned int>*                      m_optical_buffer ; 
       GSurfaceLib*                            m_basis ; 
       bool                                    m_dbgsurf ; 

   private:
       typedef std::pair<const void*, const void*> PVPV ;
       std::vector<PVPV>                       m_candidate_implicit_border_surface ;  

};

#include "GGEO_TAIL.hh"


